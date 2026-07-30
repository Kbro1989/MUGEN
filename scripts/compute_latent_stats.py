"""Precompute per-dim (K, D) mean/std of frozen-ALAE-encoder GT latents over the
train split.

The variational :class:`ALAEMotionGPT` head operates in a *standardized* latent
space so its KL prior ``N(0, I)`` is well calibrated (raw ALAE latents are not
unit-Gaussian). This script computes the standardization statistics once per K.

Run (once per K):

    conda run -n gmgpt python scripts/compute_latent_stats.py \
        --cfg configs/r6p/alae_k4_weighted_hs.yaml

Saves ``{'mean': (K, D), 'std': (K, D), 'count': int}`` to
``model.params.latent_stats_path`` (default ``checkpoints/alae_k{K}_latent_stats.pt``).
Set env ``STATS_MAX_BATCHES`` to cap the number of train batches (0 = all).

NOTE: latents are encoded from the *padded* motion exactly as in
``ALAEMotionGPT.forward_motion`` (no masking), so the statistics match the GT
latents used as the MSE-anchor / KL target during training.
"""

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Make sure imports resolve when run from anywhere (script lives in scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.archs.alae_wrapper import ALAEWrapper


@torch.no_grad()
def main():
    cfg = parse_args(phase="test")
    params = cfg.model.params

    k = int(params.k_motion_tokens)
    alae_cfg = dict(OmegaConf.to_container(params.alae, resolve=True) or {}) if params.get("alae") else {}
    alae_cfg["k"] = k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = ALAEWrapper(
        ckpt_path=params.alae_ckpt,
        alae_cfg=alae_cfg,
        unfreeze_decoder=False,
        strict_load=True,
    ).to(device)
    wrapper.eval()

    datamodule = build_data(cfg)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    out_path = params.get("latent_stats_path", None) or f"checkpoints/alae_k{k}_latent_stats.pt"
    max_batches = int(os.environ.get("STATS_MAX_BATCHES", "0"))  # 0 = all

    running_sum = None
    running_sumsq = None
    count = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        motion = batch["motion"].to(device).float()
        latents = wrapper.encode_latents(motion).float()  # (B, K, D)
        s = latents.sum(dim=0)               # (K, D)
        ss = (latents * latents).sum(dim=0)  # (K, D)
        if running_sum is None:
            running_sum, running_sumsq = s, ss
        else:
            running_sum += s
            running_sumsq += ss
        count += int(latents.shape[0])
        if (i + 1) % 50 == 0:
            print(f"[stats] batch {i + 1}, seen {count} motions")

    if count < 2:
        raise RuntimeError(f"Not enough motions to estimate latent stats (count={count}).")

    mean = running_sum / count
    var = running_sumsq / count - mean * mean
    std = var.clamp_min(0.0).sqrt()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "count": count}, out_path)
    print(f"[stats] saved (K={k}, D={mean.shape[-1]}, n={count}) -> {out_path}")
    print(
        f"[stats] per-dim std: min={std.min().item():.4f} "
        f"mean={std.mean().item():.4f} max={std.max().item():.4f}"
    )


if __name__ == "__main__":
    main()
