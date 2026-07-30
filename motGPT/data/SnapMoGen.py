"""SnapMoGen datamodule for Stage-2 (LLM + ALAE) training.

Mirrors the official snap-research/SnapMoGen ``TextMotionDataset`` recipe:
296-d features (24 joints, 30 fps), z-normalized with the official
``meta_data/mean.npy`` / ``std.npy``; per sample a caption is drawn from
manual+gpt and the motion is cropped to a ``unit_length`` multiple (capped at
``max_motion_length``) at a random offset.

Datasets emit the same 10-tuple consumed by ``humanml3d_collate``
``(text, None, None, motion, length, None, None, None, None, all_captions)``
so the whole Stage-2 pipeline (ALAEMotionGPT, MM monitor, local_eval) works
unchanged. No word_embs/pos_ohot are produced: the SnapMoGen evaluator encodes
raw caption strings with T5 (see motGPT/metrics/snapmogen.py).

Feature layout (296 = 4 + 144 + 72 + 72 + 4):
    [root(rot vel 1, lin vel 2, height 1), local 6D rotations 24*6,
     ric positions 24*3, velocities 24*3, foot contacts 4]
"""

import json
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from . import BASEDataModule
from .humanml.common.quaternion import qinv, qrot
from .utils import humanml3d_collate

SNAPMOGEN_NJOINTS = 24
SNAPMOGEN_NFEATS = 296


def recover_root_rot_pos(data):
    """Port of SnapMoGen ``utils/motion_process_bvh.py`` (same math as HML)."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang / 2, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_pos_from_ric(data, joints_num=SNAPMOGEN_NJOINTS):
    """Global joint positions from denormalized SnapMoGen features.

    Reads the explicit ric positions (dims [4+J*6 : 4+J*6+J*3]) and re-applies
    the recovered root yaw + XZ trajectory. Returns (..., J, 3).
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1 + joints_num * 6
    end_indx = start_indx + joints_num * 3
    positions = data[..., start_indx:end_indx]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return positions


class SnapMoGenText2MotionDataset(data.Dataset):
    """Official TextMotionDataset recipe, emitting humanml3d_collate tuples.

    ``name_list`` holds the clip ids and is the swap point for the datamodule's
    ``mm_mode`` (same contract as the HumanML3D datasets).
    """

    def __init__(
        self,
        data_root,
        split='train',
        mean=None,
        std=None,
        max_motion_length=320,
        min_motion_length=128,
        unit_length=8,
        tiny=False,
        pinned_list=None,
        **kwargs,
    ):
        self.max_motion_length = int(max_motion_length)
        self.min_motion_length = int(min_motion_length)
        self.unit_length = int(unit_length)

        feat_dir = pjoin(data_root, 'renamed_feats')
        split_dir = pjoin(data_root, 'data_split_info')

        if mean is None:
            mean = np.load(pjoin(data_root, 'meta_data', 'mean.npy'))
        if std is None:
            std = np.load(pjoin(data_root, 'meta_data', 'std.npy'))
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.nfeats = int(self.mean.shape[0])

        with open(pjoin(data_root, 'all_caption_clean.json'), 'r') as f:
            self.all_captions = json.load(f)

        clip_info = {}
        with open(pjoin(split_dir, f'{split}_ids.txt'), 'r') as f:
            for line in f:
                cid = line.strip()
                if not cid:
                    continue
                mid, start, end = cid.split('#')
                if int(end) - int(start) < self.min_motion_length:
                    continue
                clip_info[cid] = (mid, int(start), int(end))

        if tiny:
            clip_info = dict(list(clip_info.items())[:32])

        needed_mids = {mid for mid, _, _ in clip_info.values()}
        print(f'[SnapMoGen {split}] Loading {len(needed_mids)} feature files ...')
        self.data_dict = {}
        for mid in tqdm(sorted(needed_mids), disable=tiny):
            try:
                self.data_dict[mid] = np.load(pjoin(feat_dir, f'{mid}.npy')).astype(np.float32)
            except Exception:
                pass

        self.clip_info = {
            cid: info for cid, info in clip_info.items() if info[0] in self.data_dict
        }
        self.name_list = sorted(self.clip_info.keys())
        print(f'[SnapMoGen {split}] {len(self.name_list)} text-motion clips.')

        # ---- Optional PINNED test realization (2026-07-24) --------------------
        # The official recipe draws a random caption AND a random crop per
        # __getitem__ call, so two independent runs (ours vs a baseline) never
        # see the same prompt/window. For strictly paired comparisons a fixed
        # realization can be pinned via DATASET.SNAPMOGEN.PINNED_LIST.
        # Test split only;
        # absent/None keeps the original random behaviour byte-for-byte.
        self.pinned = None
        if pinned_list and split == 'test':
            import json as _json
            with open(pinned_list, 'r', encoding='utf-8') as _f:
                _payload = _json.load(_f)
            _items = {it['cid']: it for it in _payload['items']
                      if it['cid'] in self.clip_info}
            if not _items:
                raise RuntimeError(
                    f'PINNED_LIST {pinned_list} matched 0 of {len(self.clip_info)} clips')
            self.pinned = _items
            self.name_list = sorted(self.pinned.keys())
            _missing = len(_payload['items']) - len(_items)
            print(f'[SnapMoGen {split}] PINNED realization: {len(self.name_list)} clips '
                  f'(from {pinned_list}, {_missing} listed clips not in this split)')

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        cid = self.name_list[item]
        mid, start, end = self.clip_info[cid]
        motion = self.data_dict[mid][start:end]
        motion = (motion - self.mean) / self.std

        caps = self.all_captions[cid]
        all_captions = list(caps['manual']) + list(caps['gpt'])

        pin = self.pinned.get(cid) if getattr(self, 'pinned', None) else None
        if pin is not None:
            # Pinned realization: identical prompt/length/window across runs.
            caption = pin['caption']
            m_length = int(pin['m_length'])
            idx = int(pin['crop_start'])
        else:
            caption = random.choice(all_captions)
            m_length = min(len(motion), self.max_motion_length)
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx: idx + m_length]

        # (text, m_tokens, m_tokens_len, motion, length, word_embs, pos_ohot,
        #  text_len, tokens, all_captions, tasks, fname) -- see humanml3d_collate.
        # tasks stays None; fname carries the clip id so batch['fname'] is
        # available (e.g. for the M2T result CSV dump in local_eval.py).
        return caption, None, None, motion, m_length, None, None, None, None, all_captions, None, cid


class SnapMoGenDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        cfg.DATASET.JOINT_TYPE = 'snapmogen'
        self.name = 'snapmogen'
        self.njoints = SNAPMOGEN_NJOINTS
        self.nfeats = SNAPMOGEN_NFEATS
        cfg.DATASET.NFEATS = self.nfeats

        data_root = cfg.DATASET.SNAPMOGEN.ROOT
        self.fps = cfg.DATASET.SNAPMOGEN.FPS
        self.hparams.fps = self.fps
        self.hparams.data_root = data_root
        self.hparams.mean = np.load(pjoin(data_root, 'meta_data', 'mean.npy'))
        self.hparams.std = np.load(pjoin(data_root, 'meta_data', 'std.npy'))
        self.hparams.max_motion_length = cfg.DATASET.SNAPMOGEN.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.SNAPMOGEN.MIN_MOTION_LEN
        self.hparams.unit_length = cfg.DATASET.SNAPMOGEN.UNIT_LEN
        # Optional pinned test realization (see SnapMoGenText2MotionDataset).
        self.hparams.pinned_list = cfg.DATASET.SNAPMOGEN.get('PINNED_LIST', None)
        self.hparams.debug = cfg.DEBUG

        self.Dataset = SnapMoGenText2MotionDataset
        self.DatasetEval = SnapMoGenText2MotionDataset

    def feats2joints(self, features):
        # bf16 autocast outputs break the fp32 buffers in recover_root_rot_pos;
        # joint recovery (cumsum/trig) wants fp32 precision regardless.
        features = self.denormalize(features.float())
        return recover_pos_from_ric(features, self.njoints)

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return (features - mean) / std

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return features * std + mean

    def renorm4t2m(self, features):
        # The SnapMoGen evaluator was trained in the same normalization space
        # as the training features (official meta_data mean/std), so no
        # re-normalization is needed -- unlike HumanML3D's separate eval stats.
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(
                self.name_list, self.cfg.METRIC.MM_NUM_SAMPLES, replace=False
            )
            self.test_dataset.name_list = list(self.mm_list)
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
