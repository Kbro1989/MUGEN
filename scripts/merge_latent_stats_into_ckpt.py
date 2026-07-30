"""Merge a standalone latent-stats file into an ALAE checkpoint (format upgrade).

Produces the self-contained format read by ALAEWrapper / ALAEMotionGPT:

    {'state_dict': <ALAE weights>, 'latent_stats': {'mean', 'std', 'count'}}

Going forward alae_train.py writes this format directly; this script is the
one-time retrofit for checkpoints trained before that change.

Usage:
    conda run -n gmgpt python scripts/merge_latent_stats_into_ckpt.py \
        --ckpt  checkpoints/alae_k4.pt \
        --stats checkpoints/alae_k4_latent_stats.pt \
        --backup            # keep a weights-only copy at <ckpt>.weights_only.bak
"""
import argparse
import os
import shutil

import torch


def extract_weights(obj):
    """Return the raw ALAE state_dict from either a flat or packaged ckpt."""
    if isinstance(obj, dict) and 'state_dict' in obj and not any(
        k.startswith(('encoder_', 'decoder_', 'latent_queries')) for k in obj.keys()
    ):
        obj = obj['state_dict']
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if k != 'latent_stats'}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='ALAE weights checkpoint to upgrade')
    ap.add_argument('--stats', required=True, help='standalone latent-stats .pt (mean/std/count)')
    ap.add_argument('--out', default=None, help='output path (default: overwrite --ckpt)')
    ap.add_argument('--backup', action='store_true',
                    help='if overwriting, first copy the original to <ckpt>.weights_only.bak')
    args = ap.parse_args()
    out = args.out or args.ckpt

    weights = extract_weights(torch.load(args.ckpt, map_location='cpu'))
    stats = torch.load(args.stats, map_location='cpu')
    assert 'mean' in stats and 'std' in stats, 'stats file must contain mean/std'
    latent_stats = {
        'mean': stats['mean'].float(),
        'std': stats['std'].float(),
        'count': int(stats.get('count', 0)),
    }

    if args.backup and out == args.ckpt:
        bak = args.ckpt + '.weights_only.bak'
        if not os.path.exists(bak):
            shutil.copyfile(args.ckpt, bak)
            print(f'backup -> {bak}')

    torch.save({'state_dict': weights, 'latent_stats': latent_stats}, out)
    m = latent_stats['mean']
    print(f'merged: {len(weights)} weight tensors + latent_stats '
          f'(K={m.shape[0]}, D={m.shape[-1]}, count={latent_stats["count"]}, '
          f'std mean={latent_stats["std"].mean().item():.4f}) -> {out}')


if __name__ == '__main__':
    main()
