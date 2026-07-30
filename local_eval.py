import csv
import json
import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from motGPT.callback import build_callbacks, BestGNUCheckpoint
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def print_table(title, metrics, logger=None):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

    logger.info(metrics) if logger else None


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def save_m2t_results_csv(records, csv_path, mode, num_samples, seed, logger=None):
    """Write GNU M2T (motion->text) results to a CSV.

    Columns: ``file_name``, ``gt_text`` (all reference captions joined by " | "),
    ``pred_text`` (the model's generated caption). ``mode`` is either ``'full'``
    (write every sample) or ``'sample'`` (a seeded random subset of
    ``num_samples`` rows, so the dump stays small for spot-checking).
    """
    rows = list(records)
    total = len(rows)
    if mode == 'sample' and num_samples is not None and 0 < num_samples < total:
        rows = random.Random(seed).sample(rows, num_samples)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'gt_text', 'pred_text'])
        for r in rows:
            writer.writerow([r['fname'], r['gt_text'], r['pred_text']])

    msg = f"Saved {len(rows)}/{total} M2T results (mode={mode}) to {csv_path}"
    logger.info(msg) if logger else print(msg)


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    # TEST.CHECKPOINTS interpolates ${FOLDER} (the *experiment* dir) lazily, but
    # the next line repoints cfg.FOLDER at the test-output dir (TEST.FOLDER, e.g.
    # "results"). Resolve the checkpoint path to a concrete string here, while
    # ${FOLDER} still points at the experiment dir, so it survives the clobber.
    if cfg.TEST.CHECKPOINTS:
        cfg.TEST.CHECKPOINTS = str(cfg.TEST.CHECKPOINTS)
    cfg.FOLDER = cfg.TEST.FOLDER

    # Logger
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))

    # Output dir
    model_name = cfg.model.target.split('.')[-2].replace('_', '-').lower()
    output_dir = Path(
        os.path.join(cfg.FOLDER, model_name, cfg.NAME, "samples_" + cfg.TIME))
    if cfg.TEST.SAVE_PREDICTIONS:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to {str(output_dir)}")


    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # SnapMoGen runs differ from HumanML3D below: the BestGNU joint score uses
    # the cosine Matching direction, and visualization uses the 24-joint
    # skeleton + 30 fps (the renderer auto-picks the chain by joint count).
    is_snapmogen = 'snapmogen' in str(cfg.DATASET.target).lower()

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase="test")
    logger.info("Callbacks initialized")

    # Optional: render the full T2M test set into per-sample folders.
    # Enabled by adding `TEST.VISUALIZE: true` to the config yaml. By default
    # only 20 random samples are rendered; set `TEST.VIS_FULL: true` to render
    # the entire test set.
    if cfg.TEST.get("VISUALIZE", False):
        from motGPT.callback import T2MVisualizationCallback
        vis_dir = output_dir.parent / "visualization"
        # SnapMoGen renders its 24-joint skeleton at 30 fps; HumanML3D at 20.
        if is_snapmogen:
            fps = OmegaConf.select(cfg, "DATASET.SNAPMOGEN.FPS", default=30)
        else:
            fps = OmegaConf.select(cfg, "DATASET.HUMANML3D.FPS", default=20)
        num_workers = cfg.TEST.get("VIS_NUM_WORKERS", None)
        vis_full = bool(cfg.TEST.get("VIS_FULL", False))
        num_samples = None if vis_full else int(cfg.TEST.get("VIS_NUM_SAMPLES", 20))
        vis_seed = int(cfg.TEST.get("VIS_SEED", cfg.get("SEED_VALUE", 0)))
        # npy-only mode: with VIS_RENDER_VIDEO=false only joint .npy files are
        # written, no mp4. SAVE_FEATS=true additionally writes the renorm4t2m
        # evaluation features used for the semantic-axis scores.
        render_video = bool(cfg.TEST.get("VIS_RENDER_VIDEO", True))
        save_feats = bool(cfg.TEST.get("SAVE_FEATS", False))
        callbacks.append(
            T2MVisualizationCallback(
                output_dir=str(vis_dir),
                fps=int(fps),
                num_workers=num_workers,
                num_samples=num_samples,
                seed=vis_seed,
                logger=logger,
                render_video=render_video,
                save_feats=save_feats,
            )
        )
        logger.info(f"[T2MVis] render_video={render_video} save_feats={save_feats}")
        if vis_full:
            logger.info(f"Visualization enabled (FULL test set) -> {vis_dir}")
        else:
            logger.info(f"Visualization enabled ({num_samples} random samples, "
                        f"seed={vis_seed}) -> {vis_dir}")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, logger, phase="test")
    else:
        logger.warning("No checkpoints provided!!!")

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    task = str(OmegaConf.select(cfg, "model.params.task", default="t2m"))
    # GNU (joint T2M + M2T) checkpoints: besides the per-direction metrics the
    # model already logs (Metrics/* for T2M, Metrics/M2T_* for M2T), report the
    # same joint G+U score used to select best_gnu.ckpt during training, so the
    # test numbers are directly comparable to the checkpoint choice.
    gnu_scorer = None
    if task == 'gnu':
        gnu_scorer = BestGNUCheckpoint(
            save_dir="",  # scoring helper only; never saves checkpoints here
            div_gt=cfg.METRIC.get('DIV_GT', 9.5),
            fid_tau=cfg.METRIC.get('GNU_FID_TAU', 1.0),
            alpha=cfg.METRIC.get('GNU_ALPHA', 0.5),
            logger=logger,
            # Same direction handling as training (llm_train_alae.py): the
            # SnapMoGen Matching_score is cosine similarity, higher is better.
            matching_higher_is_better=is_snapmogen,
            # Same G-side geometric-mean weights as training selection.
            g_weights=cfg.METRIC.get('GNU_G_WEIGHTS', None),
        )

    # GNU M2T result dump: save (file_name, GT caption, predicted caption) to
    # m2t_results.csv. Configured in the yaml like TEST.VISUALIZE:
    #   TEST.M2T_SAVE_MODE: sample | full | off   (default 'sample')
    #   TEST.M2T_SAMPLE_NUM: 100                  (rows when mode == 'sample')
    #   TEST.M2T_SAMPLE_SEED: <seed>              (defaults to SEED_VALUE)
    # The model captures one row per motion during the (deterministic) non-mm
    # M2T eval pass; only the GNU task with an active M2TMetrics produces rows.
    m2t_save_mode = str(
        OmegaConf.select(cfg, "TEST.M2T_SAVE_MODE", default="sample")).lower()
    m2t_save = (task == 'gnu' and m2t_save_mode != 'off')
    m2t_sample_num = int(OmegaConf.select(cfg, "TEST.M2T_SAMPLE_NUM", default=100))
    m2t_sample_seed = int(
        OmegaConf.select(cfg, "TEST.M2T_SAMPLE_SEED", default=cfg.SEED_VALUE))
    if m2t_save:
        extra = (f", num_samples={m2t_sample_num}, seed={m2t_sample_seed}"
                 if m2t_save_mode == 'sample' else "")
        logger.info(f"M2T result CSV enabled (mode={m2t_save_mode}{extra}) "
                    f"-> {output_dir.parent / 'm2t_results.csv'}")

    # Calculate metrics
    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        # Reset the capture buffer each replication so the CSV reflects a single
        # (deterministic) pass; the mm pass below skips M2T and adds nothing.
        if m2t_save:
            model.collect_m2t_records(True)
        metrics = trainer.test(model, datamodule=datamodule)[0]
        
        stage = OmegaConf.select(cfg, "model.params.stage", default=cfg.TRAIN.STAGE)
        # Variational models fold MultiModality into the TM2TMetrics output inside
        # on_test_epoch_end (already in `metrics` and the test table above), so
        # nothing extra is needed here. Only the non-variational path needs the
        # legacy MM-noise proxy via a second mm_mode pass. This covers both the
        # pure-T2M task and the GNU joint task (whose M2T branch is skipped in
        # mm mode model-side).
        if (not getattr(model, "variational", False)
                and "TM2TMetrics" in metrics_type
                and task in ("t2m", "gnu") and stage != 'vae'):
            if hasattr(model.metrics, "MMMetrics"):
                logger.info(f"Evaluating MultiModality (mm_mode) - Replication {i}")
                datamodule.mm_mode(True)
                mm_metrics = trainer.test(model, datamodule=datamodule)[0]
                metrics.update(mm_metrics)
                datamodule.mm_mode(False)
            else:
                logger.warning(
                    "Skipping mm_mode MultiModality pass: add 'MMMetrics' to "
                    "METRIC.TYPE to enable it for this non-variational model.")
        # GNU joint score (same formula as BestGNUCheckpoint / best_gnu.ckpt):
        # Avg_G^alpha * Avg_U^(1-alpha). Computed per replication so it gets a
        # mean/conf_interval like every other metric. NB: at test BERTScore is
        # computed, so Avg_U is the 6-term mean (vs 5-term on val when
        # METRIC.M2T_BERT_ON_VAL is false).
        if gnu_scorer is not None:
            avg_g = gnu_scorer._avg_g(metrics)
            avg_u = gnu_scorer._avg_u(metrics)
            if avg_g is None or avg_u is None:
                logger.warning(
                    "GNU joint score skipped: missing T2M and/or M2T metrics "
                    "(need METRIC.TYPE: ['TM2TMetrics', 'M2TMetrics']).")
            else:
                metrics['Metrics/GNU_Avg_G'] = avg_g
                metrics['Metrics/GNU_Avg_U'] = avg_u
                metrics['Metrics/GNU_Joint_Score'] = float(
                    max(avg_g, 1e-6) ** gnu_scorer.alpha
                    * max(avg_u, 1e-6) ** (1.0 - gnu_scorer.alpha))
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {'epoch': model.epoch, 'task': model.hparams.task, 'split':model.datamodule.cfg.TEST.SPLIT}

    for key, item in all_metrics.items():
        if ('epoch' in key) or key in ['task']: continue
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval

    print_table(f"Mean Metrics", all_metrics_new, logger=logger)
    all_metrics_new.update(all_metrics)

    # Save metrics to file. output_dir itself is only created when
    # TEST.SAVE_PREDICTIONS is set, so ensure the parent exists for the metrics
    # JSON (and the M2T CSV below) regardless of that flag.
    metric_file = output_dir.parent / f"metrics_{model.hparams.task}_{model.epoch}_{all_metrics_new['split']}_{cfg.TIME}.json"
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")

    # GNU M2T result dump (see TEST.M2T_SAVE_MODE above).
    if m2t_save:
        records = getattr(model, "_m2t_records", None)
        if records:
            save_m2t_results_csv(
                records,
                output_dir.parent / "m2t_results.csv",
                m2t_save_mode,
                m2t_sample_num,
                m2t_sample_seed,
                logger=logger,
            )
        else:
            logger.warning(
                "M2T result CSV requested but no records were collected. Ensure "
                "task=gnu and 'M2TMetrics' is in METRIC.TYPE.")


if __name__ == "__main__":
    main()
