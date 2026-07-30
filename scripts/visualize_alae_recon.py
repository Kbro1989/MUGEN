"""Visualize ALAE encode -> decode reconstructions on the HumanML3D test set.

For each chosen sample, writes:
    <output_dir>/<keyid>/
        caption.txt   # text description
        gt.mp4        # ground-truth motion
        pred.mp4      # ALAE reconstruction (encode + decode)

By default renders 20 random samples; pass --full to render the entire test
set. Reuses the multiprocess-renderer infrastructure used by `local_eval.py`'s
visualization callback.

Usage:
    python scripts/visualize_alae_recon.py --cfg configs/r6p/alae_k32.yaml \\
        [--output_dir results/alae_vis] [--full] [--num_samples 20] \\
        [--num_workers 32] [--device cuda:0]
"""

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

# Silence the noisy third-party warnings (pynvml, pkg_resources, pydantic).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# PyTorch 2.6+: tolerate non-weights-only ckpts (matches the convention used
# in llm_train_alae.py).
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Make sure imports resolve when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from motGPT.config import parse_args  # noqa: E402
from motGPT.data.build_data import build_data  # noqa: E402
from motGPT.archs.alae_wrapper import ALAEWrapper  # noqa: E402
from motGPT.utils.render_utils import (  # noqa: E402
    render_fast_to_file,
    _silence_worker_warnings,
)


def _build_cfg(cfg_path: str):
    """Reuse motGPT's parse_args by forging sys.argv before invocation."""
    saved = sys.argv[:]
    sys.argv = [saved[0], "--cfg", cfg_path]
    try:
        cfg = parse_args(phase="test")
    finally:
        sys.argv = saved
    return cfg


def _resolve_keyid(fname, batch_idx: int, i: int) -> str:
    if fname:
        base = os.path.basename(str(fname))
        keyid, _ = os.path.splitext(base)
        if keyid:
            return keyid
    return f"b{batch_idx:05d}_i{i:03d}"


def _resolve_caption(text) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return "\n".join(str(x) for x in text)
    return str(text)


def _has_pair(sample_dir: str) -> bool:
    return (os.path.exists(os.path.join(sample_dir, "gt.mp4"))
            and os.path.exists(os.path.join(sample_dir, "pred.mp4")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True,
                    help="Path to an existing config yaml (e.g. configs/r6p/alae_k32.yaml). "
                         "Used only for data paths and ALAE architecture; the LM is ignored.")
    ap.add_argument("--ckpt", default=None,
                    help="Path to ALAE .pt. Defaults to cfg.ALAE_CKPT.")
    ap.add_argument("--output_dir", default="results/alae_vis",
                    help="Where to write per-sample folders.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_samples", type=int, default=20,
                    help="How many random samples to render (ignored if --full).")
    ap.add_argument("--full", action="store_true",
                    help="Render the entire chosen split instead of subsampling.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Render worker processes. Defaults to min(16, cpu//2).")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ALAE-Vis] Output -> {output_dir}")

    # Build cfg + data via the standard motGPT pipeline.
    cfg = _build_cfg(args.cfg)
    # Force test split / batch size requested.
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = args.batch_size

    datamodule = build_data(cfg)
    # Ensure datasets are constructed for the test stage.
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    if isinstance(test_loader, (list, tuple)):
        test_loader = test_loader[0]
    total = len(test_loader.dataset)
    print(f"[ALAE-Vis] {args.split} dataset size = {total}")

    # Subsample selection.
    if args.full:
        chosen = None
        print(f"[ALAE-Vis] Rendering FULL split ({total} samples).")
    else:
        rng = random.Random(args.seed)
        k = min(args.num_samples, total)
        chosen = set(rng.sample(range(total), k))
        print(f"[ALAE-Vis] Rendering {k} random samples (seed={args.seed}).")

    # Build + load ALAE (no LLM).
    alae_cfg = OmegaConf.to_container(cfg.model.params.alae, resolve=True)
    # k_motion_tokens overrides alae.k.
    alae_cfg["k"] = int(cfg.model.params.get("k_motion_tokens", alae_cfg.get("k", 64)))
    ckpt_path = args.ckpt or cfg.get("ALAE_CKPT", None)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ALAE checkpoint not found: {ckpt_path}")
    print(f"[ALAE-Vis] Loading ALAE from {ckpt_path}  k={alae_cfg['k']}")
    wrapper = ALAEWrapper(ckpt_path=ckpt_path, alae_cfg=alae_cfg,
                          unfreeze_decoder=False, strict_load=True).to(args.device).eval()

    # Process pool for rendering.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    n_workers = args.num_workers or max(1, min(16, (os.cpu_count() or 4) // 2))
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_silence_worker_warnings,
    )
    print(f"[ALAE-Vis] Render workers = {n_workers}")

    futures = []
    submitted = 0
    skipped_existing = 0
    probed = False
    global_sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="[ALAE-Vis] encode/decode")):
            motion = batch["motion"]
            lengths = batch["length"]
            texts = batch.get("text", [None] * motion.shape[0])
            fnames = batch.get("fname", [None] * motion.shape[0])

            B = int(motion.shape[0])
            # Subsample short-circuit.
            if chosen is not None:
                keep = [(global_sample_idx + i) in chosen for i in range(B)]
                if not any(keep):
                    global_sample_idx += B
                    continue
            else:
                keep = [True] * B

            motion = motion.to(args.device, non_blocking=True)
            T = int(motion.shape[1])
            # Encode + decode (frozen).
            x_hat, _, _ = wrapper.alae(motion, target_len=T)

            joints_ref = datamodule.feats2joints(motion.detach().cpu())
            joints_rst = datamodule.feats2joints(x_hat.detach().cpu())

            for i in range(B):
                if not keep[i]:
                    continue
                keyid = _resolve_keyid(fnames[i] if i < len(fnames) else None, batch_idx, i)
                sample_dir = output_dir / keyid
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Caption.
                cap_path = sample_dir / "caption.txt"
                try:
                    cap_path.write_text(_resolve_caption(texts[i] if i < len(texts) else None),
                                        encoding="utf-8")
                except Exception as exc:
                    print(f"[ALAE-Vis] caption write failed for {keyid}: {exc}")

                if _has_pair(str(sample_dir)):
                    skipped_existing += 1
                    continue

                L = max(1, int(lengths[i]))
                ref_np = joints_ref[i, :L].numpy()
                rst_np = joints_rst[i, :L].numpy()
                gt_path = str(sample_dir / "gt.mp4")
                pred_path = str(sample_dir / "pred.mp4")

                # Synchronous first render to surface worker errors early.
                if not probed:
                    probed = True
                    try:
                        if not os.path.exists(gt_path):
                            render_fast_to_file(ref_np, gt_path, args.fps)
                        if not os.path.exists(pred_path):
                            render_fast_to_file(rst_np, pred_path, args.fps)
                        print(f"[ALAE-Vis] Probe render OK ({keyid}).")
                    except Exception as exc:
                        print(f"[ALAE-Vis] Probe render FAILED ({keyid}): {exc!r}. Aborting.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        sys.exit(2)
                    continue

                if not os.path.exists(gt_path):
                    futures.append(executor.submit(render_fast_to_file, ref_np, gt_path, args.fps))
                    submitted += 1
                if not os.path.exists(pred_path):
                    futures.append(executor.submit(render_fast_to_file, rst_np, pred_path, args.fps))
                    submitted += 1

            global_sample_idx += B

    print(f"[ALAE-Vis] Encode/decode done. Waiting on {len(futures)} render jobs "
          f"(submitted={submitted}, skipped_existing={skipped_existing}) ...")

    failures = 0
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="[ALAE-Vis] rendering", unit="clip"):
        try:
            fut.result()
        except Exception as exc:
            failures += 1
            print(f"[ALAE-Vis] render job failed: {exc!r}")
    executor.shutdown(wait=True)

    print(f"[ALAE-Vis] Done. failures={failures}. Output: {output_dir}")


if __name__ == "__main__":
    main()
