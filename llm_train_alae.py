"""LLM training with a pre-trained ALAE decoder.

Pipeline (per sample):
    text + <MOT>  -> GPT2 -> K hidden states (autoregressive)
                  -> Linear projector
                  -> (frozen) ALAE decoder -> motion
Loss:
    ALAE Stage-1 reconstruction (rec + ric + perceptual + latent_l2)
    + lambda_latent_mse * MSE(predicted latents, frozen-encoder GT latents)

Usage:
    python llm_train_alae.py --cfg configs/train/alae_llm.yaml
"""

import os
# Fix for PyTorch 2.6+ checkpoint loading
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*CUDA capability.*")

import torch

# PyTorch 2.6+ safe-globals for OmegaConf checkpoints.
try:
    from omegaconf import ListConfig, DictConfig
    torch.serialization.add_safe_globals([ListConfig, DictConfig])
except Exception:
    pass

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import pytorch_lightning as pl
from omegaconf import OmegaConf

from motGPT.callback import build_callbacks, LossCSVLogger, BestFIDCheckpoint, BestGNUCheckpoint
from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained


def _summarize_trainable_params(model, logger):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameter summary: total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M "
        f"({100.0 * trainable / max(1, total):.2f}%)"
    )

    # Group counts for sanity.
    groups = {
        'alae_wrapper': 0,
        'ar_head': 0,  # projector + projector_norm + feedback_norm
        'language_model.embed': 0,
        'language_model.rest': 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('alae_wrapper'):
            groups['alae_wrapper'] += p.numel()
        elif name.startswith(('projector', 'feedback_norm')):
            groups['ar_head'] += p.numel()
        elif 'language_model' in name and ('wte' in name or 'embed' in name.lower()):
            groups['language_model.embed'] += p.numel()
        else:
            groups['language_model.rest'] += p.numel()
    for k, v in groups.items():
        logger.info(f"  trainable[{k}] = {v/1e6:.2f}M")


def main():
    torch.set_float32_matmul_precision('high')

    cfg = parse_args(phase="train")
    logger = create_logger(cfg, phase="train")
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.SEED_VALUE)

    # Loggers
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tensorboard':
            from pytorch_lightning.loggers import TensorBoardLogger
            pl_loggers.append(TensorBoardLogger(**cfg.LOGGER.TENSORBOARD.params))
    if not pl_loggers:
        from pytorch_lightning.loggers import TensorBoardLogger
        pl_loggers.append(TensorBoardLogger(save_dir=cfg.FOLDER_EXP))

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')

    # Extend the progressLogger's monitored metric list so the per-epoch
    # console line for this training script shows our ALAE-specific losses
    # in addition to validation metrics. progressLogger is created inside
    # build_callbacks() with a generic hardcoded metric_monitor; we mutate
    # it in place here to avoid touching the shared default.
    from motGPT.callback import progressLogger
    for _cb in callbacks:
        if isinstance(_cb, progressLogger):
            _cb.metric_monitor = {
                "loss": "total/train",
                "rec": "train/loss_rec",
                "ric": "train/loss_ric",
                "percept": "train/loss_percept",
                "lat_l2": "train/loss_latent_l2",
                "lat_mse": "train/loss_latent_mse",
                "FID": "Metrics/FID",
                "MMDist/gt": (
                    "Metrics/Matching_score",
                    "Metrics/gt_Matching_score",
                ),
                "Diversity/gt": (
                    "Metrics/Diversity",
                    "Metrics/gt_Diversity",
                ),
                "R@1/gt": (
                    "Metrics/R_precision_top_1",
                    "Metrics/gt_R_precision_top_1",
                ),
                "R@2/gt": (
                    "Metrics/R_precision_top_2",
                    "Metrics/gt_R_precision_top_2",
                ),
                "R@3/gt": (
                    "Metrics/R_precision_top_3",
                    "Metrics/gt_R_precision_top_3",
                ),
            }
            if cfg.model.params.get('task') == 'gnu':
                # GNU adds the motion->text (understanding) branch; surface its
                # loss + semantic metrics on the per-epoch console line.
                _cb.metric_monitor.update({
                    "m2t": "train/loss_m2t",
                    "Bleu1": "Metrics/M2T_bleu_1",
                    "Bleu4": "Metrics/M2T_bleu_4",
                    "ROUGE": "Metrics/M2T_ROUGE_L",
                    "CIDEr": "Metrics/M2T_CIDEr",
                    "Bert": "Metrics/M2T_Bert_F1",
                })
            break

    loss_csv_logger = LossCSVLogger(
        output_dir=cfg.FOLDER_EXP,
        filename="epoch_losses.csv",
    )
    callbacks.append(loss_csv_logger)
    logger.info(f"LossCSVLogger -> {cfg.FOLDER_EXP}/epoch_losses.csv")

    best_fid_callback = BestFIDCheckpoint(
        save_dir=cfg.FOLDER_EXP,
        logger=logger,
    )
    callbacks.append(best_fid_callback)
    logger.info(f"BestFIDCheckpoint -> {cfg.FOLDER_EXP}/checkpoints/best_fid.ckpt")

    # GNU (Generation + Understanding): a joint-aware best checkpoint so model
    # selection no longer ignores M2T. best_fid above stays as the generation-only
    # model; best_gnu balances both directions (see BestGNUCheckpoint).
    if cfg.model.params.get('task') == 'gnu':
        best_gnu_callback = BestGNUCheckpoint(
            save_dir=cfg.FOLDER_EXP,
            div_gt=cfg.METRIC.get('DIV_GT', 9.5),
            fid_tau=cfg.METRIC.get('GNU_FID_TAU', 1.0),
            alpha=cfg.METRIC.get('GNU_ALPHA', 0.5),
            logger=logger,
            # SnapMoGen's Matching_score is cosine similarity (higher better),
            # HumanML3D's is MMDist (lower better).
            matching_higher_is_better='snapmogen' in str(cfg.DATASET.target).lower(),
            # Geometric-mean weights over (R, FID, Match, Div) inside Avg_G.
            g_weights=cfg.METRIC.get('GNU_G_WEIGHTS', None),
        )
        callbacks.append(best_gnu_callback)
        logger.info(f"BestGNUCheckpoint -> {cfg.FOLDER_EXP}/checkpoints/best_gnu.ckpt")

    # Data
    datamodule = build_data(cfg)
    logger.info(f"DataModule loaded: {cfg.DATASET.target}")

    # Model
    model = build_model(cfg, datamodule)
    logger.info(f"Model loaded: {cfg.model.target}")

    _summarize_trainable_params(model, logger)

    # Trainer
    check_val_every = int(cfg.TRAIN.get('check_val_every_n_epoch', 5))
    train_precision = cfg.TRAIN.get('precision', 'bf16-mixed')
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=check_val_every,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true" if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        accumulate_grad_batches=cfg.TRAIN.accumulate_grad_batches,
        precision=train_precision,
        gradient_clip_val=1.0,
    )
    logger.info(
        f"Trainer ready: check_val_every_n_epoch={check_val_every}, precision={train_precision}"
    )

    if cfg.TRAIN.get('PRETRAINED', None) and not cfg.TRAIN.get('RESUME', False):
        load_pretrained(cfg, model, logger)

    if cfg.TRAIN.get('RESUME', False):
        resume_path = cfg.TRAIN.PRETRAINED
        logger.info(f"Resuming from: {resume_path}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)
    else:
        trainer.fit(model, datamodule=datamodule)

    logger.info(f"Outputs stored in: {cfg.FOLDER_EXP}")
    logger.info("Training done.")


if __name__ == "__main__":
    main()
