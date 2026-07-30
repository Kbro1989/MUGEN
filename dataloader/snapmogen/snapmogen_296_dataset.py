import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class SnapMoGenVarLenDataset(data.Dataset):
    """
    SnapMoGen variable-length dataset (296-d features, 24 joints, 30 fps).

    Loading logic mirrors the official SnapMoGen ``CommonMotionDataset``
    (snap-research/SnapMoGen, ``dataset/dataset.py``):
    - ``data_split_info/{split}_fnames.txt``: motion ids (one feature file each).
    - ``data_split_info/{split}_ids.txt``: clip ids ``mid#start#end``; clips
      shorter than ``min_len`` are dropped (official ``min_motion_length``).
    - ``renamed_feats/{mid}.npy``: per-motion feature arrays (float64 on disk,
      stored as float32 here), sliced per clip via ``[start:end]``.
    - ``meta_data/mean.npy`` / ``meta_data/std.npy``: z-normalization stats.

    The interface matches ``HumanML3DVarLenDataset``: ``__getitem__`` returns a
    z-normalized float tensor (T, 296) with a random head crop when the clip
    exceeds ``max_len``. Use with ``varlen_collate``.
    """

    def __init__(
        self,
        data_root,
        split='train',
        min_len=128,
        max_len=320,
        features_dim=296,
    ):
        self.data_root = data_root
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.features_dim = features_dim
        self.fps = 30

        self.feat_dir = pjoin(data_root, 'renamed_feats')
        meta_dir = pjoin(data_root, 'meta_data')
        split_dir = pjoin(data_root, 'data_split_info')

        try:
            mean = np.load(pjoin(meta_dir, 'mean.npy'))
            std = np.load(pjoin(meta_dir, 'std.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Error: mean.npy and std.npy not found under {meta_dir}.')

        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

        if self.mean.shape[0] != self.features_dim or self.std.shape[0] != self.features_dim:
            raise ValueError(
                f'Dimension Mismatch - Real: ({self.mean.shape[0]}), Expected: ({self.features_dim})')

        with open(pjoin(split_dir, f'{split}_fnames.txt'), 'r') as f:
            mid_list = [line.strip() for line in f if line.strip()]

        # Clip list: "mid#start#end", filtered by min_len like the official
        # min_motion_length filter.
        clips = []
        total_frames = 0
        with open(pjoin(split_dir, f'{split}_ids.txt'), 'r') as f:
            for line in f:
                cid = line.strip()
                if not cid:
                    continue
                mid, start, end = cid.split('#')
                start, end = int(start), int(end)
                if end - start < self.min_len:
                    continue
                clips.append((cid, mid, start, end))
                total_frames += end - start

        # Preload only the feature files actually referenced by kept clips
        # (official code preloads every mid in fnames.txt).
        needed_mids = {mid for _, mid, _, _ in clips}
        print(f'[SnapMoGen VarLen] Loading {split}: {len(needed_mids)}/{len(mid_list)} feature files ...')
        self.data_dict = {}
        for mid in tqdm(sorted(needed_mids)):
            try:
                self.data_dict[mid] = np.load(pjoin(self.feat_dir, f'{mid}.npy')).astype(np.float32)
            except Exception:
                pass

        self.clips = [c for c in clips if c[1] in self.data_dict]
        print(
            '[SnapMoGen VarLen] Loaded %d clips, %d frames, %.3f hours'
            % (len(self.clips), total_frames, total_frames / self.fps / 60.0 / 60.0)
        )

    def inv_transform(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        return data * self.std + self.mean

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        _, mid, start, end = self.clips[item]
        motion = self.data_dict[mid][start:end]
        total_len = len(motion)

        # Same recipe as HumanML3DVarLenDataset: keep the (near-)full clip,
        # only random-cropping the head when it exceeds max_len.
        if total_len > self.max_len:
            offset = random.randint(0, total_len - self.max_len)
            clip = motion[offset: offset + self.max_len]
        else:
            clip = motion

        clip_tensor = torch.from_numpy(np.ascontiguousarray(clip)).float()
        clip_normalized = (clip_tensor - self.mean) / self.std
        return clip_normalized
