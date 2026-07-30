#!/usr/bin/env python
"""Download the complete SnapMoGen dataset from HuggingFace.

The SnapMoGen dataset (https://huggingface.co/datasets/Ericguo5513/SnapMoGen,
https://github.com/snap-research/SnapMoGen) is ~16.5 GB and contains:

    renamed_feats.zip      (~12.9 GB)  extracted motion features (pickle)
    renamed_bvhs.zip       (~3.5  GB)  raw BVH motion-capture files
    all_caption_clean.json (~75   MB)  text captions
    data_split_info/                   train / val / test id + fname lists
    meta_data/             mean.npy, std.npy  (feature normalisation stats)
    codes/                 reference loading / processing scripts
    README.md, LICENSE, teaser.png

Usage
-----
    python download_snap_dataset.py --save_dir /path/to/data

This downloads everything into ``<save_dir>/SnapMoGen`` (matching the layout the
official repo expects: ``cp -r ./SnapMoGen your_data_folder/SnapMoGen``). If
``--save_dir`` already ends in ``SnapMoGen`` it is used as-is.

The download is resumable: re-running the command continues where it left off
and skips files that are already complete (verified by hash/size).

Optionally unzip the two archives after download:

    python download_snap_dataset.py --save_dir /path/to/data --extract

Note: extracting roughly doubles the on-disk footprint (to ~33 GB). Add
``--remove_archives`` to delete the .zip files once they are extracted.
"""

import argparse
import os
import shutil
import sys
import zipfile

ARCHIVES = ["renamed_feats.zip", "renamed_bvhs.zip"]


def human(n_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.2f} PB"


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def resolve_target(save_dir):
    """Return the directory the dataset should live in (<save_dir>/SnapMoGen)."""
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    if os.path.basename(save_dir.rstrip("/")) == "SnapMoGen":
        return save_dir
    return os.path.join(save_dir, "SnapMoGen")


def check_free_space(target, needed_bytes):
    # walk up to the first existing parent to stat the filesystem
    probe = target
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free
    print(f"  free space on target filesystem: {human(free)}")
    if free < needed_bytes:
        print(
            f"  WARNING: only {human(free)} free but ~{human(needed_bytes)} is "
            "recommended. Download may fail partway."
        )
    return free


def download(repo_id, target, workers, token):
    from huggingface_hub import snapshot_download

    print(f"Downloading '{repo_id}' (dataset) -> {target}")
    print(f"  workers={workers}  (resumable: already-complete files are skipped)")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        max_workers=workers,
        token=token,
        # show per-file tqdm bars
    )
    return path


def relocate_archives(target, archive_dir):
    """Move the .zip archives out of ``target`` into ``archive_dir``.

    Used to park the archives on a roomier (possibly slower) filesystem before
    extracting into ``target`` (a smaller/faster one). Moving the zips off
    ``target`` first frees that space, so extraction can write the contents
    back with no double-counting. Returns nothing; archives end up in
    ``archive_dir``.
    """
    os.makedirs(archive_dir, exist_ok=True)
    for name in ARCHIVES:
        src = os.path.join(target, name)
        dst = os.path.join(archive_dir, name)
        if os.path.exists(dst) and not os.path.exists(src):
            print(f"  [ok] {name} already in {archive_dir}")
            continue
        if not os.path.exists(src):
            print(f"  [skip] {name} not found in {target}")
            continue
        print(f"Moving {name} ({human(os.path.getsize(src))}) -> {archive_dir} ...")
        shutil.move(src, dst)  # cross-filesystem: copy then unlink
        print(f"  moved (freed that space on {target}'s filesystem)")


def extract_archives(target, archive_dir=None, remove_archives=False):
    """Extract the archives into ``target``.

    Archives are read from ``archive_dir`` if given (e.g. after relocating),
    otherwise from ``target`` itself.
    """
    src_dir = archive_dir or target
    for name in ARCHIVES:
        zip_path = os.path.join(src_dir, name)
        if not os.path.exists(zip_path):
            print(f"  [skip] {name} not found in {src_dir}")
            continue
        print(f"Extracting {name} ({human(os.path.getsize(zip_path))}) -> {target} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        print(f"  done: {name}")
        if remove_archives:
            os.remove(zip_path)
            print(f"  removed archive {name}")


def verify(repo_id, target, token, skip=()):
    """Compare local files against the remote file list.

    Filenames in ``skip`` are not expected on disk (e.g. archives that were
    intentionally removed after extraction) and are excluded from the check.
    """
    from huggingface_hub import HfApi

    skip = set(skip)
    api = HfApi()
    info = api.repo_info(
        repo_id, repo_type="dataset", files_metadata=True, token=token
    )
    checked = [s for s in info.siblings if s.rfilename not in skip]
    missing, ok = [], 0
    for s in checked:
        local = os.path.join(target, s.rfilename)
        if not os.path.exists(local):
            missing.append(s.rfilename)
            continue
        if s.size is not None and os.path.getsize(local) != s.size:
            missing.append(f"{s.rfilename} (size mismatch)")
            continue
        ok += 1
    print(f"\nVerification: {ok}/{len(checked)} files present and correct.")
    if skip:
        print(f"  (skipped {len(skip)} removed archive(s): {', '.join(sorted(skip))})")
    if missing:
        print("  MISSING / INCOMPLETE:")
        for m in missing:
            print(f"    - {m}")
        return False
    print(f"  Total on disk: {human(dir_size(target))}")
    print("  All files verified. ✔")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Download the complete SnapMoGen dataset from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--save_dir",
        required=True,
        help="Parent data folder; dataset is placed in <save_dir>/SnapMoGen.",
    )
    p.add_argument(
        "--repo_id",
        default="Ericguo5513/SnapMoGen",
        help="HuggingFace dataset repo id.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (only needed if the repo becomes gated).",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="Unzip renamed_feats.zip and renamed_bvhs.zip after download.",
    )
    p.add_argument(
        "--archive_dir",
        default=None,
        help=(
            "Move the .zip archives to this dir before extracting (e.g. a "
            "roomier filesystem). Extraction reads from here and writes the "
            "contents into <save_dir>/SnapMoGen. The archives are kept here."
        ),
    )
    p.add_argument(
        "--remove_archives",
        action="store_true",
        help="Delete the .zip files after extracting (use with --extract).",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip the download step (files already present); just extract/verify.",
    )
    p.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip the post-download verification step.",
    )
    args = p.parse_args()

    # Faster transfer if hf_transfer is installed; harmless otherwise.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    target = resolve_target(args.save_dir)
    os.makedirs(target, exist_ok=True)
    print(f"Target directory: {target}")

    archive_dir = (
        os.path.abspath(os.path.expanduser(args.archive_dir))
        if args.archive_dir
        else None
    )

    # Extracting in-place needs room for download + extracted (~35 GB). With an
    # --archive_dir on another FS, the zips are moved off target first, so the
    # extracted contents roughly replace them (~17 GB).
    if args.extract and not archive_dir:
        needed = 35 * 1024**3
    else:
        needed = 18 * 1024**3
    check_free_space(target, needed)

    if not args.skip_download:
        download(args.repo_id, target, args.workers, args.token)

    if args.extract:
        if archive_dir:
            relocate_archives(target, archive_dir)
        extract_archives(target, archive_dir=archive_dir,
                         remove_archives=args.remove_archives)

    if not args.skip_verify:
        # Archives are no longer under target if moved away or removed.
        skip = ARCHIVES if (args.extract and (archive_dir or args.remove_archives)) else ()
        ok = verify(args.repo_id, target, args.token, skip=skip)
        if not ok:
            print("\nSome files are missing — re-run the same command to resume.")
            sys.exit(1)

    print(f"\nDone. SnapMoGen dataset is at: {target}")


if __name__ == "__main__":
    main()
