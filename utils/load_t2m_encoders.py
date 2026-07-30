"""Frozen T2M motion-encoder loader for use as perceptual loss.

Wraps the same ``MovementConvEncoder`` + ``MotionEncoderBiGRUCo`` pair that
``motGPT.metrics.t2m.TM2TMetrics._get_t2m_evaluator`` instantiates for
R-Precision / FID evaluation. The returned modules are ``.eval()`` and have
``requires_grad=False`` on every parameter so they can be safely plugged into
the AE training loop without polluting the optimizer.

Usage
-----
>>> move_enc, motion_enc = load_t2m_motion_encoders(device)
>>> # x: (B, T, 263) z-score-normalised HumanML3D features
>>> # lengths: (B,) original (un-padded) frame counts
>>> feat = motion_enc(move_enc(x[..., :259]), (lengths // 4).clamp_min(1))
>>> feat.shape  # (B, 512)
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from motGPT.archs.tm2t_evaluator import MovementConvEncoder, MotionEncoderBiGRUCo

# Defaults match configs/evaluator/tm2t.yaml for HumanML3D (NFEATS=263).
DEFAULT_T2M_CKPT = os.path.join(
    'deps', 't2m', 't2m', 'text_mot_match', 'model', 'finest.tar'
)
DEFAULT_MOVE_INPUT_SIZE = 259  # NFEATS(263) - 4 foot-contact dims
DEFAULT_MOVE_HIDDEN = 512
DEFAULT_MOVE_OUTPUT = 512
DEFAULT_MOTION_HIDDEN = 1024
DEFAULT_MOTION_OUTPUT = 512


def load_t2m_motion_encoders(
    device: torch.device | str = 'cpu',
    checkpoint_path: str = DEFAULT_T2M_CKPT,
    move_input_size: int = DEFAULT_MOVE_INPUT_SIZE,
    move_hidden: int = DEFAULT_MOVE_HIDDEN,
    move_output: int = DEFAULT_MOVE_OUTPUT,
    motion_hidden: int = DEFAULT_MOTION_HIDDEN,
    motion_output: int = DEFAULT_MOTION_OUTPUT,
) -> Tuple[MovementConvEncoder, MotionEncoderBiGRUCo]:
    """Load the frozen HumanML3D T2M ``movement`` and ``motion`` encoders.

    The encoders are returned in ``eval()`` mode with all parameters'
    ``requires_grad=False``. They live in fp32 on ``device``.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'T2M evaluator checkpoint not found at {checkpoint_path!r}. '
            'Run prepare/download_t2m_evaluators.sh first.'
        )

    move_enc = MovementConvEncoder(move_input_size, move_hidden, move_output)
    motion_enc = MotionEncoderBiGRUCo(move_output, motion_hidden, motion_output)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    move_enc.load_state_dict(ckpt['movement_encoder'])
    motion_enc.load_state_dict(ckpt['motion_encoder'])

    for enc in (move_enc, motion_enc):
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        enc.to(device)

    return move_enc, motion_enc


def compute_t2m_perceptual_features(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    move_enc: MovementConvEncoder,
    motion_enc: MotionEncoderBiGRUCo,
    feature_slice: slice = slice(0, 259),
) -> torch.Tensor:
    """Encode a (B, T, 263) batch into a (B, 512) clip embedding.

    The encoders are run in fp32; the caller is responsible for ensuring the
    surrounding autocast context (if any) is disabled.
    """
    # Drop foot-contact dims so the input matches MovementConvEncoder.input_size.
    feats = motion[..., feature_slice].float()
    # Two stride-2 convs => temporal downsampling by 4.
    lens_ds = (lengths // 4).clamp_min(1)
    movement = move_enc(feats)  # (B, T/4, 512)
    # MotionEncoderBiGRUCo uses pack_padded_sequence internally; it needs
    # cap_lens as a 1-D long tensor or list. Clip so it never exceeds the
    # actual downsampled time dimension produced by the conv.
    lens_ds = lens_ds.clamp_max(movement.shape[1])
    # cuDNN RNN backward refuses to run when the GRU module is in eval mode.
    # We keep params frozen but disable cuDNN for this call so backward can
    # propagate through the GRU to give the AE a perceptual gradient signal.
    with torch.backends.cudnn.flags(enabled=False):
        return motion_enc(movement, lens_ds)
