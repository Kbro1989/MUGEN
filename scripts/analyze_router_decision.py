"""Extract Weighted-HS router decisions over a full split (READ-ONLY analysis).

For every text in the eval split, runs the WHS model's routed AR rollout
(`_encode_text_ar`) in eval mode and records the per-slot routing over the L
GPT-2 layers: raw logits (B, K, L) captured by a forward hook on
``layer_router`` plus the deterministic tempered-softmax weights the model
actually used (``_last_router_weights``, tau = ``_router_infer_tau()``,
i.e. the converged tau_end=0.5 unless ``router_eval_tau`` overrides).

Usage (needs a GPU; L4 is plenty -- prompt forward only, no decoding/eval):

    python scripts/analyze_router_decision.py --cfg configs/server/alae_k4_nll_whs_eval.yaml

Env overrides:
    RTR_CKPT=/path/to/epoch=NNN.ckpt   analyze a training snapshot instead of
                                       the frozen best_fid (routing evolution)
    RTR_OUT=/path/out_root             output root (default: results/router_analysis)
    RTR_MAX_BATCHES=N                  stop after N batches (smoke test)

Outputs under <out_root>/<cfg.NAME>_<epoch>_<split>/:
    router_decisions.npz   weights (N,K,L) f32, logits (N,K,L) f32, tau
    meta.json              texts (order-aligned), ckpt path, split, epoch
Visualization/statistics live in scripts/plot_router_decision.py.
"""
import json
import os
import sys

# Runnable as `python scripts/analyze_router_decision.py` from anywhere: put
# the repo root (this file's parent's parent) on sys.path for motGPT imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytorch_lightning as pl
import torch
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def main():
    cfg = parse_args(phase="test")
    if cfg.TEST.CHECKPOINTS:
        cfg.TEST.CHECKPOINTS = str(cfg.TEST.CHECKPOINTS)
    ckpt_override = os.environ.get('RTR_CKPT')
    if ckpt_override:
        cfg.TEST.CHECKPOINTS = ckpt_override
    cfg.FOLDER = cfg.TEST.FOLDER
    out_root = os.environ.get(
        'RTR_OUT', 'results/router_analysis')

    logger = create_logger(cfg, phase="test")
    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)
    assert hasattr(model, 'layer_router'), (
        'cfg.model.target must be the WeightedHS class; got '
        f'{cfg.model.target}')
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)
    load_pretrained(cfg, model, logger, phase="test")
    pl.seed_everything(cfg.SEED_VALUE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    cap = {}
    model.layer_router.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__('logits', out.detach().float().cpu()))

    try:
        datamodule.setup('test')
    except TypeError:
        datamodule.setup()
    loader = datamodule.test_dataloader()

    max_batches = int(os.environ.get('RTR_MAX_BATCHES', 0))
    weights, logits, texts, names = [], [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                print(f'RTR_MAX_BATCHES={max_batches} reached, stopping early')
                break
            bt = [str(t) for t in batch['text']]
            model._encode_text_ar(bt)
            weights.append(model._last_router_weights.float().cpu().numpy())
            # R2 composes logits AFTER layer_router (static + bounded delta);
            # prefer its _last_router_logits over the raw hook capture.
            composed = getattr(model, '_last_router_logits', None)
            if composed is not None:
                logits.append(composed.float().cpu().numpy())
            else:
                logits.append(cap['logits'].numpy())
            texts.extend(bt)
            for key in ('fname', 'name', 'fnames', 'names'):
                if key in batch:
                    names.extend([str(x) for x in batch[key]])
                    break
            if bi % 20 == 0:
                print(f'batch {bi}: {len(texts)} texts done', flush=True)

    W = np.concatenate(weights)   # (N, K, L)
    G = np.concatenate(logits)    # (N, K, L)
    assert W.shape == G.shape and W.shape[0] == len(texts), (W.shape, G.shape, len(texts))
    tau = float(model._router_infer_tau())
    epoch = getattr(model, 'epoch', 'na')

    split = str(cfg.TEST.SPLIT)
    outdir = os.path.join(out_root, f'{cfg.NAME}_ep{epoch}_{split}')
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, 'router_decisions.npz'),
                        weights=W, logits=G, tau=tau)
    with open(os.path.join(outdir, 'meta.json'), 'w') as f:
        json.dump({'texts': texts, 'names': names, 'tau': tau,
                   'ckpt': str(cfg.TEST.CHECKPOINTS), 'split': split,
                   'epoch': str(epoch), 'shape': list(W.shape)}, f)

    print(f'saved {W.shape} -> {outdir}')
    print('global mean routing weights (K x L):')
    print(np.array2string(W.mean(axis=0), precision=3, suppress_small=True))


if __name__ == '__main__':
    main()
