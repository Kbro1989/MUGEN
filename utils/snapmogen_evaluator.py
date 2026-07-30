"""Frozen SnapMoGen text-motion evaluator (TMR-style) loader.

Ports the official evaluator from snap-research/SnapMoGen
(``model/evaluator/modules.py`` + ``evaluator_wrapper.py``): an ACTOR-style
transformer VAE encoder over the first 148 motion dims (root info + 24x6D
local rotations) and a matching text tower fed by frozen T5-v1_1-base token
embeddings. Weights are the official release checkpoint
(``net_best_top1.tar``), fetched by ``prepare/download_snapmogen_evaluator.sh``
into ``deps/snapmogen/evaluator/``.

Protocol notes (these DIFFER from the HumanML3D TM2T evaluator):
- FID / Diversity are computed on ``fid_emb`` (the raw first-token encoder
  output, before the mu/logvar linear); R-precision / Matching use the VAE
  mean ``mu`` -- two different embeddings.
- Retrieval similarity is cosine; the matching score is mean cosine
  similarity, so HIGHER is better (opposite direction of HML MMDist).
- Motion inputs must be z-normalized with the official SnapMoGen
  ``meta_data/mean.npy`` / ``std.npy`` (``SnapMoGenVarLenDataset`` already
  does this); the wrapper slices ``[..., :148]`` internally so the full 296-d
  features can be passed straight in.

Usage
-----
>>> evaluator = load_snapmogen_evaluator(device)
>>> # motion: (B, T, 296) z-normalized SnapMoGen features, lengths: (B,)
>>> fid_emb, mu = evaluator.encode_motion(motion, lengths)   # (B,256),(B,256)
>>> text_mu = evaluator.encode_text(['a person walks'])      # (B, 256)
>>> percept = compute_snapmogen_perceptual_features(motion, lengths, evaluator)
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_EVALUATOR_DIR = os.path.join(
	'deps', 'snapmogen', 'evaluator', 'eval_klde-5_late-5_nlayer6_norm'
)
# Official evaluator.yaml hyperparameters (fixed by the released checkpoint).
DIM_POSE = 148           # 1+2+1+24*6: root info + local 6D rotations
LATENT_DIM = 256
FF_SIZE = 1024
NUM_LAYERS = 6
NUM_HEADS = 4
DROPOUT = 0.1
ACTIVATION = 'gelu'
T5_VERSION = 'google/t5-v1_1-base'
T5_DIM = 768
MAX_TEXT_LENGTH = 120


def length_to_mask(length, max_len, device: torch.device = None) -> Tensor:
	if device is None:
		device = 'cpu'
	if isinstance(length, list):
		length = torch.tensor(length)
	length = length.to(device)
	mask = torch.arange(max_len, device=device).expand(
		len(length), max_len
	) < length.unsqueeze(1)
	return mask


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

	def forward(self, x: Tensor) -> Tensor:
		# batch_first: x is (B, T, D)
		x = x + self.pe[:, : x.shape[1], :]
		return self.dropout(x)


class SnapMoGenVAEEncoder(nn.Module):
	"""ACTOR-style transformer VAE encoder (official ``Encoder``, vae=True).

	Two learnable distribution tokens are prepended; ``forward`` returns the
	raw first-token output (``fid_emb``) plus the linear-projected (mu, logvar)
	token pair.
	"""

	def __init__(
		self,
		nfeats: int,
		latent_dim: int = LATENT_DIM,
		ff_size: int = FF_SIZE,
		num_layers: int = NUM_LAYERS,
		num_heads: int = NUM_HEADS,
		dropout: float = DROPOUT,
		activation: str = ACTIVATION,
	) -> None:
		super().__init__()
		self.nbtokens = 2  # mu / logvar (vae=True in the released checkpoint)
		self.projection = nn.Linear(nfeats, latent_dim)
		self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
		self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=latent_dim,
			nhead=num_heads,
			dim_feedforward=ff_size,
			dropout=dropout,
			activation=activation,
			batch_first=True,
		)
		self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.linear = nn.Linear(in_features=latent_dim, out_features=latent_dim)

	def forward(self, x_dict: Dict) -> Tensor:
		x = self.projection(x_dict['x'])
		mask = x_dict['mask']
		bs = len(x)
		tokens = self.tokens[None].expand(bs, -1, -1)
		xseq = torch.cat((tokens, x), 1)
		token_mask = torch.ones((bs, self.nbtokens), dtype=torch.bool, device=x.device)
		aug_mask = torch.cat((token_mask, mask), 1).bool()
		xseq = self.sequence_pos_encoding(xseq)
		final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
		return final[:, 0], self.linear(final[:, : self.nbtokens])

	def encode(self, input, mask, sample_mean=True):
		fid_emb, output = self.forward({'x': input, 'mask': mask})
		mu, logvar = output.unbind(1)
		logvar = torch.clamp(logvar, -10.0, 10.0)
		if sample_mean:
			return_vec = mu
		else:
			std = logvar.mul(0.5).exp()
			return_vec = mu + std * torch.randn_like(std)
		return fid_emb, return_vec, (mu, logvar)


class SnapMoGenEvaluator(nn.Module):
	"""Frozen two-tower evaluator with the official metric embeddings.

	``encode_motion``  -> ``(fid_emb, mu)``; gradients flow through to the
	motion input (params frozen), so ``fid_emb`` doubles as a perceptual
	feature for AE training.
	``encode_text``    -> ``mu`` (no_grad; T5 is lazy-loaded on first call so
	training-only use never pays for it).
	"""

	def __init__(self, max_text_length: int = MAX_TEXT_LENGTH, t5_version: str = T5_VERSION):
		super().__init__()
		self.latent_enc = SnapMoGenVAEEncoder(nfeats=DIM_POSE)
		self.text_enc = SnapMoGenVAEEncoder(nfeats=T5_DIM)
		self.max_text_length = max_text_length
		self.t5_version = t5_version
		# Plain-dict holder so the (frozen, ~250M-param) T5 never registers as a
		# submodule: this evaluator may itself live inside a checkpointed module
		# (e.g. the Stage-2 metrics), and T5 must stay out of its state_dict.
		self._t5 = {}

	@property
	def device(self):
		return next(self.latent_enc.parameters()).device

	def _ensure_t5(self):
		if not self._t5:
			from transformers import AutoTokenizer, T5EncoderModel

			self._t5['tokenizer'] = AutoTokenizer.from_pretrained(self.t5_version, legacy=False)
			model = T5EncoderModel.from_pretrained(self.t5_version).eval()
			for p in model.parameters():
				p.requires_grad = False
			self._t5['model'] = model.to(self.device)
		elif next(self._t5['model'].parameters()).device != self.device:
			# The host module may have been moved after the lazy load.
			self._t5['model'] = self._t5['model'].to(self.device)

	def encode_motion(self, motion: Tensor, lengths, sample_mean: bool = True):
		"""motion: (B, T, >=148) z-normalized feats; lengths: (B,) valid frames."""
		feats = motion[..., :DIM_POSE].float()
		mask = length_to_mask(lengths, feats.shape[1], feats.device)
		fid_emb, return_vec, _ = self.latent_enc.encode(feats, mask, sample_mean=sample_mean)
		return fid_emb, return_vec

	@torch.no_grad()
	def encode_text(self, texts, sample_mean: bool = True) -> Tensor:
		self._ensure_t5()
		tokens = self._t5['tokenizer'](
			list(texts),
			max_length=self.max_text_length,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			add_special_tokens=True,
			return_tensors='pt',
		)
		input_ids = tokens['input_ids'].to(self.device)
		attention_mask = tokens['attention_mask'].to(self.device)
		embeddings = self._t5['model'](
			input_ids=input_ids, attention_mask=attention_mask
		)['last_hidden_state']
		_, return_vec, _ = self.text_enc.encode(
			embeddings, attention_mask.bool(), sample_mean=sample_mean
		)
		return return_vec


def load_snapmogen_evaluator(
	device: torch.device | str = 'cpu',
	ckpt_dir: str = DEFAULT_EVALUATOR_DIR,
	model_name: str = 'net_best_top1',
) -> SnapMoGenEvaluator:
	"""Load the frozen SnapMoGen evaluator (both towers, eval mode, no grads)."""
	ckpt_path = os.path.join(ckpt_dir, 'model', f'{model_name}.tar')
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(
			f'SnapMoGen evaluator checkpoint not found at {ckpt_path!r}. '
			'Run prepare/download_snapmogen_evaluator.sh first.'
		)
	weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)

	evaluator = SnapMoGenEvaluator()
	evaluator.latent_enc.load_state_dict(weights['latent_enc'])
	evaluator.text_enc.load_state_dict(weights['text_enc'])
	evaluator.eval()
	for p in evaluator.parameters():
		p.requires_grad = False
	evaluator.to(device)
	print(f"Loaded SnapMoGen evaluator from epoch {weights.get('ep', '?')} ({ckpt_path})")
	return evaluator


def compute_snapmogen_perceptual_features(
	motion: Tensor,
	lengths,
	evaluator: SnapMoGenEvaluator,
) -> Tensor:
	"""Encode a (B, T, 296) batch into the (B, 256) FID embedding.

	Uses ``fid_emb`` so the perceptual loss optimizes the exact feature space
	the SnapMoGen FID metric is computed in. The evaluator is frozen but
	differentiable w.r.t. the motion input; run in fp32 (caller disables
	autocast, matching the T2M perceptual path).
	"""
	fid_emb, _ = evaluator.encode_motion(motion, lengths, sample_mean=True)
	return fid_emb
