"""Wrapper around the trained ALAE (Adaptive-Length AutoEncoder).

Used by downstream LLM training: the LLM produces K hidden states per sample,
which are projected into the ALAE latent space and decoded back to motion
features by the (typically frozen) ALAE decoder.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from adaptive_length_auto_encoder import AdaptiveLengthAutoEncoder


def _build_alae_from_cfg(alae_cfg: dict) -> AdaptiveLengthAutoEncoder:
	"""Instantiate ALAE with the hyperparameters that match the saved ckpt."""
	defaults = dict(
		input_dim=263,
		k=64,
		latent_dim=512,
		hidden_dim=512,
		depth=3,
		dilation_growth_rate=3,
		activation='gelu',
		norm=None,
		num_res_blocks=2,
		num_encoder_layers=4,
		num_decoder_layers=4,
		nhead=8,
		dim_feedforward=2048,
		dropout=0.05,
		max_decode_len=256,
	)
	merged = {**defaults, **(alae_cfg or {})}
	return AdaptiveLengthAutoEncoder(**merged)


class ALAEWrapper(nn.Module):
	"""Holds an ``AdaptiveLengthAutoEncoder`` plus freeze/unfreeze controls.

	The encoder is always frozen (only used to produce GT latents for an
	auxiliary latent-MSE loss). The decoder is frozen by default but can be
	thawed via ``unfreeze_decoder=True`` for downstream fine-tuning.
	"""

	def __init__(
		self,
		ckpt_path: str,
		alae_cfg: Optional[dict] = None,
		unfreeze_decoder: bool = False,
		strict_load: bool = True,
	):
		super().__init__()
		self.alae = _build_alae_from_cfg(alae_cfg or {})
		self.unfreeze_decoder = bool(unfreeze_decoder)
		self.ckpt_path = ckpt_path

		# Latent standardization stats for the variational LLM head, if the ckpt
		# is the self-contained format. None for legacy weights-only ckpts.
		self.latent_stats = None

		if ckpt_path and os.path.isfile(ckpt_path):
			state = torch.load(ckpt_path, map_location='cpu')
			if isinstance(state, dict):
				# Self-contained ALAE ckpt carries per-dim latent mean/std next to
				# the weights (written by alae_train.py); read it before unwrapping.
				ls = state.get('latent_stats', None)
				if isinstance(ls, dict) and 'mean' in ls and 'std' in ls:
					self.latent_stats = {
						'mean': ls['mean'], 'std': ls['std'],
						'count': ls.get('count', None),
					}
				# Unwrap a packaged state_dict (only when the top level isn't
				# itself raw ALAE weights), then drop any non-weight meta keys.
				if 'state_dict' in state and not any(
					k.startswith(('encoder_', 'decoder_', 'latent_queries')) for k in state.keys()
				):
					state = state['state_dict']
				state = {k: v for k, v in state.items() if k != 'latent_stats'}
			missing, unexpected = self.alae.load_state_dict(state, strict=False)
			if strict_load and (missing or unexpected):
				raise RuntimeError(
					f'ALAE ckpt mismatch. missing={missing[:5]} unexpected={unexpected[:5]}'
				)
		elif ckpt_path:
			raise FileNotFoundError(f'ALAE checkpoint not found: {ckpt_path}')

		# Always freeze encoder side; decoder per switch.
		self._set_encoder_grad(False)
		self._set_decoder_grad(self.unfreeze_decoder)

	# ------------------------------------------------------------------
	# Freezing helpers
	# ------------------------------------------------------------------
	def _encoder_modules(self):
		return [
			self.alae.encoder_backbone,
			self.alae.encoder_blocks,
			self.alae.encoder_norm,
		]

	def _decoder_modules(self):
		return [
			self.alae.decoder_blocks,
			self.alae.decoder_norm,
			self.alae.decoder_refiner,
		]

	def _set_encoder_grad(self, requires_grad: bool):
		for mod in self._encoder_modules():
			for p in mod.parameters():
				p.requires_grad_(requires_grad)
		# latent_queries belong to the encoder side.
		self.alae.latent_queries.requires_grad_(requires_grad)

	def _set_decoder_grad(self, requires_grad: bool):
		for mod in self._decoder_modules():
			for p in mod.parameters():
				p.requires_grad_(requires_grad)

	def set_decoder_trainable(self, trainable: bool):
		"""Public switch to thaw / freeze the ALAE decoder at any time."""
		self.unfreeze_decoder = bool(trainable)
		self._set_decoder_grad(self.unfreeze_decoder)

	# ------------------------------------------------------------------
	# Convenience accessors
	# ------------------------------------------------------------------
	@property
	def k(self) -> int:
		return int(self.alae.k)

	@property
	def latent_dim(self) -> int:
		return int(self.alae.latent_dim)

	@property
	def input_dim(self) -> int:
		return int(self.alae.input_dim)

	# ------------------------------------------------------------------
	# Forward helpers
	# ------------------------------------------------------------------
	@torch.no_grad()
	def encode_latents(self, motion: torch.Tensor) -> torch.Tensor:
		"""Return frozen GT latents ``[B, K, latent_dim]``."""
		was_training = self.alae.training
		self.alae.eval()
		latents, _ = self.alae.encode(motion)
		if was_training:
			self.alae.train()
		return latents

	def decode_to_motion(self, latents: torch.Tensor, target_len: int) -> torch.Tensor:
		"""Decode latents into motion features ``[B, target_len, input_dim]``."""
		if self.unfreeze_decoder:
			return self.alae.decode(latents, target_len=target_len)
		# Frozen decoder: keep BN/Dropout in eval and disable any internal grad
		# update, but still allow gradients to flow back to ``latents``.
		was_training = self.alae.training
		self.alae.eval()
		try:
			out = self.alae.decode(latents, target_len=target_len)
		finally:
			if was_training:
				self.alae.train()
		return out

	def compute_loss(
		self,
		x: torch.Tensor,
		x_hat: torch.Tensor,
		latents: torch.Tensor,
		mask: Optional[torch.Tensor] = None,
		lengths: Optional[torch.Tensor] = None,
		t2m_move_enc=None,
		t2m_motion_enc=None,
		percept_fn=None,
		lambda_rec: float = 1.0,
		lambda_ric: float = 0.5,
		lambda_percept: float = 0.5,
		lambda_latent_l2: float = 1e-4,
		ric_slice: slice = slice(4, 67),
		use_percept_cosine: bool = False,
		lambda_percept_cosine: float = 1.0,
	):
		return self.alae.compute_loss(
			x,
			x_hat,
			latents,
			mask=mask,
			lengths=lengths,
			t2m_move_enc=t2m_move_enc,
			t2m_motion_enc=t2m_motion_enc,
			percept_fn=percept_fn,
			lambda_rec=lambda_rec,
			lambda_ric=lambda_ric,
			lambda_percept=lambda_percept,
			lambda_latent_l2=lambda_latent_l2,
			ric_slice=ric_slice,
			use_percept_cosine=use_percept_cosine,
			lambda_percept_cosine=lambda_percept_cosine,
		)
