"""Text-to-motion via LLM hidden states + frozen ALAE decoder (pure AR head).

Pipeline:
  text + <MOT>  -> GPT2 -> h_0
  h_{t-1}       -> GPT2 (with KV cache, optional) -> h_t  for t in 1..K-1
  [h_0, ..., h_{K-1}] -> LayerNorm -> Linear projector -> (mu, logvar)

The K latents are decoded by the (optionally fine-tuned) ALAE decoder into
the HumanML3D 263-d motion features.

Losses:
	- ALAE reconstruction loss (rec + ric + perceptual + latent_l2), same as Stage 1.
		- Auxiliary set-aware mean anchor vs. frozen ALAE encoder GT latents.
	- NLL recipe (latent_loss_mode 'nll' / 'chamfer_nll', 2026-07-03): the anchor
	  becomes a Gaussian NLL so sigma learns the per-text conditional spread,
	  the per-sample losses decode mu (never a sampled z), and the FID/R eval
	  pass samples z = mu + sigma*eps (METRIC.FID_SAMPLE_TEMP 1.0).
"""

from __future__ import annotations

import json
import os
from os.path import join as pjoin
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motGPT.config import get_obj_from_str, instantiate_from_config
from motGPT.models.base import BaseModel
from motGPT.archs.alae_wrapper import ALAEWrapper

# Perceptual encoders are loaded lazily to avoid mandatory T2M ckpt at import.


def _build_inputs(texts: List[str], mot_token_str: str) -> List[str]:
	"""Build the per-sample LLM input string ending in the <MOT> seed token."""
	return [f'Generate motion: {t.strip()} {mot_token_str}' for t in texts]


class ALAEMotionGPT(BaseModel):
	"""LLM + ALAE-decoder text-to-motion model (pure AR rollout head).

	Key options
	-----------
	use_kv_cache: bool
		If True, accelerate the K-step AR rollout with HF GPT2 ``past_key_values``.
		Default False.
	unfreeze_alae_decoder: bool
		If True, unfreeze the ALAE decoder for joint fine-tuning. Default False.
	"""

	def __init__(
		self,
		cfg,
		datamodule,
		lm,
		alae_ckpt: str,
		alae: Optional[dict] = None,
		stage: str = 'lm_pretrain',
		condition: str = 'text',
		task: str = 't2m',
		metrics_dict=('TM2TMetrics',),
		guidance_scale: float = 1.0,
		use_kv_cache: bool = False,
		unfreeze_alae_decoder: bool = False,
		freeze_text_embeddings: bool = True,
		k_motion_tokens: int = 64,
		lambda_rec: float = 1.0,
		lambda_ric: float = 0.5,
		lambda_percept: float = 0.5,
		lambda_latent_l2: float = 1e-4,
		lambda_latent_mse: float = 0.1,
		use_percept_cosine: bool = False,
		lambda_percept_cosine: float = 1.0,
		t2m_perceptual_ckpt: str = 'deps/t2m/t2m/text_mot_match/model/finest.tar',
		# SnapMoGen replaces the HumanML3D T2M encoder pair with the official
		# SnapMoGen evaluator (motion tower) for the perceptual loss; selected
		# automatically when the datamodule is snapmogen.
		snapmogen_evaluator_dir: str = 'deps/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm',
		disable_perceptual: bool = False,
		# ----- Variational latent head (generative sampling for real MM) -----
		# The head outputs (mu, logvar) over a *standardized* latent space;
		# training samples z~N(mu,sigma^2) (reparam) and adds a free-bits KL. This
		# is now the only supported motion-generation path so MultiModality comes
		# from the same generator rather than from a deterministic fallback.
		variational: bool = True,
		lambda_kl: float = 1e-2,
		kl_warmup_epochs: int = 30,
		kl_free_bits: float = 0.0,
		latent_stats_path: Optional[str] = None,
		eval_sample_temperature: float = 1.0,
		# Mean anchor for the latent distribution. ``fixed`` is the old slot-wise
		# MSE; ``chamfer`` / ``sinkhorn`` treat the K latents as a set, matching the
		# ALAE decoder's near permutation-invariant memory use. ``fixed_chamfer`` /
		# ``fixed_sinkhorn`` keep the old anchor and add a set term.
		latent_loss_mode: str = 'chamfer',
		lambda_latent_set: float = 1.0,
		latent_sinkhorn_epsilon: float = 0.05,
		latent_sinkhorn_iters: int = 30,
		# ----- Cross-slot AR latent sampling (2026-07-03) -----
		# The independent per-slot diagonal Gaussian ignores the measured
		# cross-slot residual correlation (|rho| 0.34-0.52 at K=4; residual
		# effective dim ~63/2048), which floors the sampled FID (~34) far above
		# the oracle (1.28) -- correlation-intact-but-text-shuffled residuals
		# score FID 19.3, i.e. the correlation STRUCTURE matters more than
		# per-dim scale. This flag factorizes the joint by the chain rule:
		#   p(z_1..z_K|text) = prod_k p(z_k | text, z_<k)
		# Train: ONE teacher-forced forward -- GT latents z_gt_{<k} are embedded
		# (latent_in_projector) and appended after the <MOT> seed; slot k's
		# (mu_k, logvar_k) is read at position k-1 and scored with per-step
		# Gaussian NLL (= exact chain-rule likelihood of the GT set).
		# Eval: sequential rollout -- SAMPLE z_k, feed it back, predict k+1;
		# the joint now carries the cross-slot correlations. Requires
		# latent_loss_mode 'nll' (chain fixes slot order; set matching would
		# scramble the conditioning). K=1 degenerates to the plain NLL head.
		ar_latent_sampling: bool = False,
		# ----- Low-rank factor head (2026-07-03 late) -----
		# rank r of the text-conditioned low-rank + diagonal covariance over the
		# flattened K*D latent (0 = off). Single-shot structured sampling: the
		# global basis U captures within-slot AND cross-slot correlation with no
		# sequential feedback (no exposure bias). Requires latent_loss_mode
		# 'nll'; exclusive with ar_latent_sampling. r ~ 64 = measured residual
		# effective dim.
		latent_low_rank: int = 0,
		# ----- Global set-level MDN mixture head (2026-07-06) -----
		# M > 0 upgrades the single low-rank+diagonal Gaussian to a GLOBAL
		# M-component mixture over the flattened K*D latent:
		#   q(z|text) = sum_m pi_m(text) * N(z; mu + U c_m(text), Sigma)
		# with Sigma = U diag(a)^2 U^T + diag(sigma^2) TIED across components
		# and component centers restricted to span(U) (c_m in R^r): the
		# measured residual effective dim (~63/2048) says mode separation
		# lives in the factor subspace, and the 2026-07-06 oracle verdict
		# (K8 sampled asymptote 24.5 vs oracle 6.89 @ s=0.5, capacity
		# monotone in K) pins the remaining FID gap on the SINGLE-MODE
		# family, not on mu. Tied covariance structurally rules out the
		# classic MDN sigma-collapse and per-component variance competition;
		# the objective is exact ML (logsumexp over M shared-Cholesky
		# Woodbury quadratics), NO bound. Training keeps the decoupled
		# contract: mu's gradient stays bitwise independent of
		# (pi, c, a, U, sigma); the mixture does ML on the mu-detached
		# residual. c = 0 degenerates EXACTLY to _factor_gaussian_nll for
		# any M and pi. Requires latent_low_rank > 0.
		# 0 = off (default): no new parameters, no behavior change.
		latent_mdn_components: int = 0,
		# Init std of the component-center bias (factor-coefficient space).
		# Breaks the component permutation symmetry (identical components get
		# identical gradients and never split); the per-dim perturbation is
		# O(1e-3) vs the early residual, so training still starts as the
		# validated single-Gaussian factor head (the bias=-2 philosophy).
		# Read only when latent_mdn_components > 0.
		latent_mdn_offset_init: float = 0.1,
		# Usage load-balance weight (KL of batch-mean posterior usage to
		# uniform), added on the mixture side only. 0 = pure exact ML
		# (default). Rescue knob for responsibility starvation -- engage
		# ~0.01 in a FRESH run only on sentinel evidence (mdn_use_min -> 0).
		latent_mdn_balance: float = 0.0,
		# ----- Scheduled sampling on the AR context (2026-07-03, same day) -----
		# Pure GT teacher forcing COLLAPSED at eval (K4 run: train rec/percept
		# 0.138/0.115 -- 3x below any prior run, because the GT prefix slots
		# leak the answer through the rho~0.5 cross-slot correlation -- while
		# the free-running chain scored R@1 0.30 / FID ~55; the rollout-vs-TF
		# consistency check passed at 5e-6, so this is a train/eval CONTEXT
		# distribution gap, not a plumbing bug). Fix: two-pass training. Pass 1
		# (no grad) runs the GT-context chain and draws each slot from its own
		# predictive distribution z~_k = mu_k + sigma_k*eps; pass 2 trains the
		# NLL of the GT slots given that PERTURBED context. The conditionals
		# learn to error-correct an imperfect prefix (contraction instead of
		# compounding), and the per-sample losses re-engage the text (a noisy
		# context can no longer carry the answer). Self-annealing: early on the
		# context is near-marginal noise (the chain starts text-driven, like
		# K=1), tightening as mu improves. Costs one extra no-grad LM forward.
		ar_context_sampling: bool = False,
		# ----- GNU phase (Generation + Understanding: joint T2M + M2T) -----
		# Enabled when ``task == 'gnu'``. Adds a motion->text (understanding)
		# branch on top of the existing T2M path; each train step optimizes both
		# (loss = L_t2m + lambda_m2t * L_m2t). Text embeddings stay frozen; the
		# new motion-input projector + <MOT> marker are the trainable additions.
		lambda_m2t: float = 1.0,
		m2t_max_new_tokens: int = 40,
		# Label smoothing on the M2T cross-entropy. Combats the over-confidence /
		# exposure-bias that makes free-running captions degrade while the
		# teacher-forced CE keeps dropping. 0.0 == plain CE (backward compatible).
		m2t_label_smoothing: float = 0.0,
		# Train-only Gaussian noise on the M2T input latents, scaled per dim by
		# the latent_std buffer (fraction-of-natural-spread semantics). The M2T
		# input is otherwise a deterministic function of the motion (frozen
		# encoder), so GPT-2 can key memorized captions on exact latent values;
		# label smoothing alone slows but does not stop the resulting
		# val-caption decay. 0.0 == off (backward compatible).
		m2t_latent_noise_std: float = 0.0,
		# Flooding level b on the M2T CE (Ishida et al. 2020: |CE-b|+b). The
		# caption task's intrinsic conditional entropy (one-to-many motion->text,
		# under label smoothing 0.1) puts a *generalizing* CE near ~2.8; in every
		# run the val-caption decay began exactly when train CE crossed below
		# ~2.6-2.8, because going lower is only achievable by memorizing which
		# exact caption/words a training sample uses. The floor forbids that
		# region while corrective descent resumes automatically whenever
		# T2M-driven feature drift pushes CE back above b. 0.0 == off.
		m2t_loss_floor: float = 0.0,
		# Beam width for M2T eval decoding (1 == greedy). >1 changes the eval
		# protocol -- use only for explicit decoding comparisons.
		m2t_num_beams: int = 1,
		debug: bool = False,
		**kwargs,
	):
		self.save_hyperparameters(ignore='datamodule', logger=False)
		self.datamodule = datamodule
		self.njoints = self.datamodule.njoints
		self.fps = self.datamodule.fps
		super().__init__()

		# ----- LLM -----
		self.lm = instantiate_from_config(lm)
		self.tokenizer = self.lm.tokenizer
		self.language_model = self.lm.language_model
		hidden_size = self.language_model.config.hidden_size

		# ----- ALAE (frozen by default; decoder optionally trainable) -----
		alae_cfg = dict(alae) if alae is not None else {}
		# k_motion_tokens is the single source of truth for K; override any
		# stale `k` that might be present in the alae sub-config.
		if 'k' in alae_cfg and int(alae_cfg['k']) != int(k_motion_tokens):
			print(
				f"[ALAEMotionGPT] overriding alae.k={alae_cfg['k']} with "
				f"k_motion_tokens={k_motion_tokens}"
			)
		alae_cfg['k'] = int(k_motion_tokens)
		self.alae_wrapper = ALAEWrapper(
			ckpt_path=alae_ckpt,
			alae_cfg=alae_cfg,
			unfreeze_decoder=unfreeze_alae_decoder,
			strict_load=True,
		)
		if self.alae_wrapper.k != int(k_motion_tokens):
			raise ValueError(
				f'k_motion_tokens={k_motion_tokens} but ALAE wrapper has k={self.alae_wrapper.k}'
			)

		if not bool(variational):
			raise ValueError(
				'ALAEMotionGPT now requires variational=True; deterministic '
				'non-VAE motion generation was removed so MM is generated by the '
				'latent distribution.'
			)

		# ----- AR head: per-step Linear from LLM hidden to ALAE latent params,
		# with a LayerNorm on the hidden states fed back as inputs_embeds. -----
		self.variational = True
		self.projector_norm = nn.LayerNorm(hidden_size)
		proj_out = 2 * self.alae_wrapper.latent_dim
		self.projector = nn.Linear(hidden_size, proj_out)
		self.feedback_norm = nn.LayerNorm(hidden_size)

		# ----- Cross-slot AR latent sampling (see the ar_latent_sampling doc
		# in __init__). The feedback token is the (standardized) LATENT value,
		# not the hidden state: GT z at train (teacher forcing), sampled z at
		# eval -- that value-feedback is what carries cross-slot correlation
		# into the chain. Linear + LayerNorm mirrors the feedback_norm pattern.
		self.ar_latent_sampling = bool(ar_latent_sampling)
		if self.ar_latent_sampling:
			mode_norm = str(latent_loss_mode).lower().replace('-', '_')
			if mode_norm != 'nll':
				raise ValueError(
					"ar_latent_sampling requires latent_loss_mode 'nll': the "
					"chain rule fixes slot order, so set-matching anchors "
					f"(got '{latent_loss_mode}') would scramble the conditioning.")
			self.latent_in_projector = nn.Linear(self.alae_wrapper.latent_dim, hidden_size)
			self.latent_in_norm = nn.LayerNorm(hidden_size)

		# ----- Low-rank factor head (single-shot structured covariance) -----
		# q(z|text) = N(mu, U diag(a(text))^2 U^T + diag(sigma(text)^2)) over the
		# FLATTENED K*D latent: the learned global basis U spans slots, so ONE
		# draw carries both within-slot and cross-slot correlation -- no
		# sequential feedback, hence no exposure bias by construction (the
		# continuous AR chain collapsed without scheduled sampling and drifted
		# with it, while the single-shot diagonal head was stable but
		# structure-blind at FID ~26). rank ~64 matches the measured residual
		# effective dim (63/2048). NLL is closed-form via Woodbury + the matrix
		# determinant lemma; a(text) starts small (bias -2) so training begins
		# as the proven diagonal model and grows factor structure smoothly.
		self.latent_low_rank = int(latent_low_rank)
		if self.latent_low_rank > 0:
			mode_norm = str(latent_loss_mode).lower().replace('-', '_')
			if mode_norm not in ('nll', 'chamfer_nll'):
				raise ValueError(
					"latent_low_rank requires latent_loss_mode 'nll' or "
					f"'chamfer_nll' (got '{latent_loss_mode}').")
			if self.ar_latent_sampling:
				raise ValueError(
					'latent_low_rank and ar_latent_sampling are exclusive: the '
					'factor head is the single-shot alternative to the chain.')
			kd = int(k_motion_tokens) * self.alae_wrapper.latent_dim
			u0 = torch.randn(kd, self.latent_low_rank)
			u0 = u0 / u0.norm(dim=0, keepdim=True)  # unit-norm directions
			self.latent_factors = nn.Parameter(u0)
			self.factor_scale_head = nn.Linear(hidden_size, self.latent_low_rank)
			with torch.no_grad():
				self.factor_scale_head.bias.fill_(-2.0)  # a ~= e^-2: near-diagonal start

		# ----- Global set-level MDN mixture head (see the constructor doc) -----
		self.latent_mdn_components = int(latent_mdn_components)
		self._mdn_train_stats = None  # sentinel hand-off (plain attr, never in state_dict)
		if self.latent_mdn_components > 0:
			if self.latent_low_rank <= 0:
				raise ValueError(
					'latent_mdn_components requires latent_low_rank > 0: the '
					'mixture shares the factor basis U (centers U c_m live in '
					'span(U)) and its covariance IS the factor covariance.')
			if self.ar_latent_sampling:
				raise ValueError(
					'latent_mdn_components and ar_latent_sampling are '
					'exclusive: the mixture is the single-shot alternative '
					'to the chain.')
			m_comp = self.latent_mdn_components
			# pi logits: zero weights + zero bias -> exactly uniform pi for
			# every text at the start; every component receives gradient from
			# step 0 (no starvation at birth).
			self.mdn_logit_head = nn.Linear(hidden_size, m_comp)
			# Component centers in factor-coefficient space (Delta_m = U c_m).
			# Zero weights + small random bias = tiny DISTINCT text-independent
			# seed offsets: symmetry breaking without leaving the validated
			# single-Gaussian start (see latent_mdn_offset_init doc).
			self.mdn_center_head = nn.Linear(
				hidden_size, m_comp * self.latent_low_rank)
			with torch.no_grad():
				self.mdn_logit_head.weight.zero_()
				self.mdn_logit_head.bias.zero_()
				self.mdn_center_head.weight.zero_()
				self.mdn_center_head.bias.normal_(
					0.0, float(latent_mdn_offset_init))

		# ----- Special <MOT> token -----
		self.mot_token_str = '<MOT>'
		self._original_vocab_size = self.language_model.get_input_embeddings().weight.shape[0]
		added = self.tokenizer.add_special_tokens({'additional_special_tokens': [self.mot_token_str]})
		if added > 0:
			self.language_model.resize_token_embeddings(len(self.tokenizer))
			with torch.no_grad():
				emb = self.language_model.get_input_embeddings().weight
				existing = emb[: self._original_vocab_size].float()
				mean_val = existing.mean().item()
				std_val = existing.std().item()
				new_init = torch.randn(
					emb.shape[0] - self._original_vocab_size,
					emb.shape[1],
					device=emb.device,
					dtype=torch.float32,
				) * std_val + mean_val
				emb[self._original_vocab_size:] = new_init.to(emb.dtype)
			# The new <MOT> row must be trainable even when the backbone is
			# otherwise frozen (LoRA freezes every base weight, embeddings
			# included). Flip the embedding (and untied output head) back on;
			# the freeze hook below restricts gradients to the new row(s) when
			# freeze_text_embeddings is set, so only <MOT> actually updates.
			in_emb = self.language_model.get_input_embeddings()
			in_emb.weight.requires_grad_(True)
			out_head = self.language_model.get_output_embeddings()
			if out_head is not None and out_head.weight is not in_emb.weight:
				out_head.weight.requires_grad_(True)
		self.mot_token_id = self.tokenizer.convert_tokens_to_ids(self.mot_token_str)

		# Freeze the original text-embedding rows via a gradient hook (full LLM
		# fine-tuning otherwise enabled).
		if freeze_text_embeddings:
			self._install_text_embedding_freeze_hook()

		# ----- Perceptual encoders (frozen, lazy) -----
		self._t2m_move_enc = None
		self._t2m_motion_enc = None
		self._disable_perceptual = bool(disable_perceptual)
		self._t2m_perceptual_ckpt = t2m_perceptual_ckpt
		# Dataset-dependent plumbing: SnapMoGen swaps the perceptual encoder for
		# the official SnapMoGen evaluator and supervises the explicit-position
		# slice [0:148] (root + 6D rotations) instead of HumanML3D's ric [4:67].
		self._is_snapmogen = getattr(datamodule, 'name', '') == 'snapmogen'
		self._ric_slice = slice(0, 148) if self._is_snapmogen else slice(4, 67)
		self._snap_percept_evaluator = None
		self._snapmogen_evaluator_dir = snapmogen_evaluator_dir

		# ----- Misc plumbing for BaseModel -----
		self.guidance_scale = guidance_scale
		self.feats2joints = datamodule.feats2joints
		self.render_videos = False
		self.vis_num = 0

		# Opt-in per-sample M2T result capture (consumed by local_eval.py to write
		# m2t_results.csv). None == disabled; set to a list via
		# ``collect_m2t_records`` to accumulate {fname, gt_text, pred_text} rows
		# during the GNU M2T eval passes. Plain-T2M checkpoints never touch it.
		self._m2t_records = None

		# ----- Latent standardization buffers -----
		# The variational head works in a standardized latent space so the KL
		# prior N(0, I) is well calibrated. Stats are precomputed offline by
		# scripts/compute_latent_stats.py; registered as buffers so they move
		# with the module and round-trip through the checkpoint.
		latent_dim = self.alae_wrapper.latent_dim
		K = int(k_motion_tokens)
		mean = torch.zeros(K, latent_dim)
		std = torch.ones(K, latent_dim)
		# Primary source: stats embedded in the self-contained ALAE ckpt
		# (written by alae_train.py). Fallback: a standalone latent_stats_path
		# file (legacy / retrofit via scripts/compute_latent_stats.py).
		stats = getattr(self.alae_wrapper, 'latent_stats', None)
		src = f'ALAE ckpt ({alae_ckpt})' if stats is not None else None
		if stats is None and latent_stats_path and os.path.isfile(latent_stats_path):
			stats = torch.load(latent_stats_path, map_location='cpu')
			src = latent_stats_path
		if stats is not None:
			mean = stats['mean'].float().reshape(K, latent_dim)
			std = stats['std'].float().reshape(K, latent_dim).clamp_min(1e-4)
			print(f'[ALAEMotionGPT] loaded latent stats from {src} '
			      f'(std mean={std.mean().item():.4f})')
		else:
			print(f'[ALAEMotionGPT] WARNING: no latent stats embedded in the ALAE '
			      f'ckpt and latent_stats_path not found ({latent_stats_path}); '
			      f'using mean=0/std=1 — embed stats via alae_train.py or run '
			      f'scripts/compute_latent_stats.py or the KL prior will be '
			      f'miscalibrated.')
		self.register_buffer('latent_mean', mean)
		self.register_buffer('latent_std', std)

		# ----- GNU (Generation + Understanding) motion-understanding branch -----
		# Only built for the GNU phase, so plain-T2M checkpoints keep an identical
		# parameter set. The motion-input projector maps frozen ALAE encoder
		# latents into GPT-2's embedding space so a motion can be fed to the LLM
		# as K continuous "motion tokens" for the M2T (motion->text) direction.
		self.is_gnu = (str(task) == 'gnu')
		if self.is_gnu:
			self.motion_in_projector = nn.Linear(self.alae_wrapper.latent_dim, hidden_size)
			# Fixed M2T instruction prefix; tokenized once and kept as a buffer so
			# it rides along with the module / device. Same string for every
			# sample, so no padding is ever needed on the prompt side.
			self._m2t_prefix = 'Please describe the following human motion using plain text:'
			prefix_ids = self.tokenizer(
				self._m2t_prefix, return_tensors='pt', add_special_tokens=False
			)['input_ids'][0]
			self.register_buffer('_m2t_prefix_ids', prefix_ids, persistent=False)
			# GPT-2's config sets no loss_type, so HF logs a one-time warning the
			# first time forward_m2t passes labels ("loss_type=None unrecognised,
			# using ForCausalLMLoss"). That default IS the causal-LM loss we want;
			# set it explicitly to silence the (harmless) warning.
			try:
				self.language_model.config.loss_type = 'ForCausalLMLoss'
			except Exception:
				pass

	# ----------------------------------------------------------------------
	# Base hooks (used by motGPT BaseModel)
	# ----------------------------------------------------------------------
	def loss_log_dict(self, split: str):  # override: we log losses via self.log
		return {}

	def configure_metrics(self):
		# Defer to BaseMetrics like BaseModel does, but tolerate missing keys.
		from motGPT.metrics import BaseMetrics
		self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

	def configure_optimizers(self):
		"""Discriminative fine-tuning LRs, opt-in via ``TRAIN.OPTIM.backbone_lr``.

		The pretrained GPT-2 blocks get ``backbone_lr`` while the freshly
		initialized modules (AR-head projector + norms, motion_in_projector,
		token embeddings -- only the <MOT> row receives gradients) keep the base
		``params.lr``. A full-rate backbone keeps drifting on T2M long after M2T
		peaks (the GNU timescale mismatch), and 1e-4 is high for fine-tuning a
		124M LM in the first place.

		Also switches to selective weight decay: LN/bias and the wte/wpe
		embedding tensors are excluded. The embedding exclusion is load-bearing,
		not just convention -- AdamW's decoupled decay shrinks the WHOLE wte
		tensor every step, including the hook-frozen text-embedding rows, so
		with decay on them "frozen" rows still drift toward 0.

		Without ``backbone_lr`` in the config this defers to the base
		single-group optimizer, so pre-GNU configs reproduce exactly.
		"""
		optim_cfg = self.hparams.cfg.TRAIN.OPTIM
		backbone_lr = optim_cfg.get('backbone_lr', None)
		if backbone_lr is None:
			return super().configure_optimizers()
		backbone_lr = float(backbone_lr)

		optim_target = optim_cfg.target
		if len(optim_target.split('.')) == 1:
			optim_target = 'torch.optim.' + optim_target
		extra = dict(optim_cfg.params)
		head_lr = float(extra.pop('lr'))
		weight_decay = float(extra.pop('weight_decay', 0.0))
		# Co-trained ALAE decoder (present only when ``unfreeze_alae_decoder``
		# thawed it) gets its own, gentler LR. The decoder is pretrained and is
		# the component producing the FID-1.4 Stage-1 reconstruction; the 1e-4
		# head rate would wreck it. Defaults to ``head_lr`` if unset (so a thawed
		# decoder without ``decoder_lr`` still trains, just not gently).
		decoder_lr = optim_cfg.get('decoder_lr', None)
		decoder_lr = head_lr if decoder_lr is None else float(decoder_lr)

		def _is_embedding_name(name):
			# GPT-2 embeddings are named wte/wpe; Llama/Qwen-family backbones
			# name theirs embed_tokens (+ an untied lm_head when present). The
			# decay exclusion below is load-bearing for EVERY backbone -- AdamW's
			# decoupled decay shrinks the WHOLE embedding tensor each step, so
			# hook-frozen text rows would still drift toward 0 (2026-07-23 qwen3
			# fix; GPT-2 runs are byte-identical: their param names never
			# contain the new keys).
			return any(k in name for k in ('wte', 'wpe', 'embed_tokens', 'lm_head'))

		def tier(name):
			# Only the (optionally) thawed ALAE decoder is trainable inside the
			# wrapper -- the encoder + latent_queries stay frozen -- so any
			# requires_grad ``alae_wrapper`` param is a decoder param.
			if 'alae_wrapper' in name:
				return 'decoder'
			# NB: substring match, not startswith -- the GPT-2 params are reached
			# first through self.lm, so deduplicated names read 'lm.language_model.*'.
			if 'language_model' in name and not _is_embedding_name(name):
				return 'backbone'
			return 'head'

		lr_map = {'decoder': decoder_lr, 'backbone': backbone_lr, 'head': head_lr}

		def is_no_decay(name, p):
			return p.ndim < 2 or _is_embedding_name(name)

		buckets = {}
		for name, p in self.named_parameters():
			if not p.requires_grad:
				continue
			buckets.setdefault((tier(name), is_no_decay(name, p)), []).append(p)
		param_groups = [
			{'params': ps,
			 'lr': lr_map[t],
			 'weight_decay': 0.0 if nd else weight_decay}
			for (t, nd), ps in buckets.items()
		]

		def _tier_numel(target):
			return sum(p.numel() for (t, _), ps in buckets.items()
			           if t == target for p in ps)
		n_bb, n_hd, n_dec = (_tier_numel('backbone'),
		                     _tier_numel('head'), _tier_numel('decoder'))
		print(f'[ALAEMotionGPT] discriminative LRs: backbone {n_bb / 1e6:.2f}M '
		      f'@ {backbone_lr:g}, heads {n_hd / 1e6:.2f}M @ {head_lr:g}, '
		      f'decoder {n_dec / 1e6:.2f}M @ {decoder_lr:g} '
		      f'(wd={weight_decay:g}, LN/bias/emb excluded)')

		optimizer = get_obj_from_str(optim_target)(params=param_groups, **extra)
		return {'optimizer': optimizer,
		        'lr_scheduler': self._build_lr_scheduler(optimizer)}

	# ----------------------------------------------------------------------
	# Embedding freezing
	# ----------------------------------------------------------------------
	def _install_text_embedding_freeze_hook(self):
		embed = self.language_model.get_input_embeddings()
		orig_n = int(self._original_vocab_size)

		def _hook(grad):
			if grad is None:
				return grad
			grad = grad.clone()
			grad[:orig_n].zero_()
			return grad

		embed.weight.register_hook(_hook)

		# Output head (lm_head) typically ties to input embedding for GPT2 — if
		# untied, freeze its rows too.
		out_head = self.language_model.get_output_embeddings()
		if out_head is not None and out_head.weight is not embed.weight:
			out_head.weight.register_hook(_hook)

	# ----------------------------------------------------------------------
	# Perceptual encoder lazy load
	# ----------------------------------------------------------------------
	def _ensure_t2m_encoders(self):
		if self._disable_perceptual:
			return
		if self._is_snapmogen:
			if self._snap_percept_evaluator is None:
				from utils.snapmogen_evaluator import load_snapmogen_evaluator
				self._snap_percept_evaluator = load_snapmogen_evaluator(
					device=self.device,
					ckpt_dir=self._snapmogen_evaluator_dir,
				)
			return
		if self._t2m_move_enc is not None:
			return
		from utils.load_t2m_encoders import load_t2m_motion_encoders
		move_enc, motion_enc = load_t2m_motion_encoders(
			device=self.device,
			checkpoint_path=self._t2m_perceptual_ckpt,
		)
		self._t2m_move_enc = move_enc
		self._t2m_motion_enc = motion_enc

	def _percept_fn(self):
		"""Dataset-appropriate perceptual feature callable (or None)."""
		if self._snap_percept_evaluator is None:
			return None
		from utils.snapmogen_evaluator import compute_snapmogen_perceptual_features
		evaluator = self._snap_percept_evaluator
		return lambda motion, lengths: compute_snapmogen_perceptual_features(
			motion, lengths, evaluator
		)

	def _drop_t2m_encoder_state(self, state_dict):
		return {
			k: v for k, v in state_dict.items()
			if not k.startswith((
				'_t2m_move_enc.', '_t2m_motion_enc.', '_snap_percept_evaluator.',
			))
		}

	def state_dict(self, *args, **kwargs):
		state = super().state_dict(*args, **kwargs)
		return self._drop_t2m_encoder_state(state)

	def preprocess_state_dict(self, state_dict):
		filtered_state_dict = self._drop_t2m_encoder_state(state_dict)
		return super().preprocess_state_dict(filtered_state_dict)

	# ----------------------------------------------------------------------
	# Forward: text -> latent distribution params (B, K, D_alae)
	# ----------------------------------------------------------------------
	def _compute_latent_dist(self, texts: List[str],
	                         teacher_latents: Optional[torch.Tensor] = None):
		"""AR head -> ``(mu, logvar)``.

		``mu``/``logvar`` live in the *standardized* latent space and are split
		from the 2*D projector output.

		``teacher_latents`` (B, K, D, standardized): when given and
		``ar_latent_sampling`` is on, the K per-slot distributions are computed
		with the GT latents teacher-forced as the chain context (one forward);
		slot k's params are then conditioned on the REAL z_{<k}.
		"""
		if self.ar_latent_sampling and teacher_latents is not None:
			hidden = self._encode_text_ar_teacher_forced(texts, teacher_latents)
		else:
			hidden = self._encode_text_ar(texts)
		out = self.projector(self.projector_norm(hidden))
		mu, logvar = out.chunk(2, dim=-1)
		return mu, logvar

	def _compute_latent_dist_factor(self, texts: List[str]):
		"""Factor head -> ``(mu, logvar, log_a, log_pi, c)``.

		``mu``/``logvar`` (B, K, D) as in the diagonal head; ``log_a`` (B, r)
		are the text-conditioned log-amplitudes of the r global covariance
		factors (read from the slot-pooled normed hidden). When the MDN head
		is on, ``log_pi`` (B, M) and ``c`` (B, M, r) are the mixture
		log-weights and component centers (factor-coefficient space);
		otherwise both are None.
		"""
		hidden = self._encode_text_ar(texts)
		h = self.projector_norm(hidden)
		out = self.projector(h)
		mu, logvar = out.chunk(2, dim=-1)
		pooled = h.mean(dim=1)
		log_a = self.factor_scale_head(pooled)  # (B, r)
		log_pi = c = None
		if self.latent_mdn_components > 0:
			log_pi = F.log_softmax(self.mdn_logit_head(pooled), dim=-1)
			# Blow-up guard in the spirit of the +-10 logvar clamps; healthy
			# per-coordinate c is O(1), so this is inert in practice.
			c = self.mdn_center_head(pooled).view(
				mu.shape[0], self.latent_mdn_components, self.latent_low_rank
			).clamp(-16.0, 16.0)
		return mu, logvar, log_a, log_pi, c

	def _factor_gaussian_nll(self, mu: torch.Tensor, logvar: torch.Tensor,
	                         log_a: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""Per-dim NLL under ``N(mu, U diag(a)^2 U^T + diag(sigma^2))``.

		Woodbury + matrix determinant lemma over the flattened K*D latent:
		  Sigma^-1 = D^-1 - D^-1 F (I + F^T D^-1 F)^-1 F^T D^-1,  F = U diag(a)
		  log|Sigma| = log|I + F^T D^-1 F| + sum log sigma^2
		Only an (r x r) Cholesky per sample. Scaled per-dim (mean over K*D and
		batch) so ``lambda_latent_mse`` transfers from the diagonal NLL; with
		a -> 0 this reduces exactly to ``_gaussian_nll``.
		"""
		B, K, D = mu.shape
		kd = K * D
		resid = (target - mu).reshape(B, kd)
		lv = logvar.clamp(-8.0, 8.0).reshape(B, kd)
		dinv = torch.exp(-lv)                                  # (B, KD)
		a = torch.exp(log_a.clamp(-6.0, 4.0))                  # (B, r)
		u = self.latent_factors                                # (KD, r)
		ud = u.unsqueeze(0) * dinv.unsqueeze(-1)               # (B, KD, r) = D^-1 U
		utdu = torch.einsum('dr,bds->brs', u, ud)              # (B, r, r) = U^T D^-1 U
		m = a.unsqueeze(-1) * utdu * a.unsqueeze(-2)
		m = m + torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
		chol = torch.linalg.cholesky(m)
		w = torch.einsum('bd,bdr->br', resid, ud) * a          # F^T D^-1 resid
		minv_w = torch.cholesky_solve(w.unsqueeze(-1), chol).squeeze(-1)
		quad = (resid.pow(2) * dinv).sum(-1) - (w * minv_w).sum(-1)
		logdet = lv.sum(-1) + 2.0 * torch.log(
			torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
		return (0.5 * (quad + logdet) / kd).mean()

	def _mdn_mixture_nll(self, mu: torch.Tensor, logvar: torch.Tensor,
	                     log_a: torch.Tensor, log_pi: torch.Tensor,
	                     c: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""Per-dim NLL of the tied-covariance global mixture (EXACT ML).

		q(z) = sum_m pi_m N(z; mu + U c_m, Sigma) with Sigma = U diag(a)^2 U^T
		+ diag(sigma^2) SHARED across components -> one Woodbury factorization
		per sample; per-component work is r-dimensional only:
		  quad_m = (resid - U c_m)^T Sigma^-1 (resid - U c_m)
		         = q0 - 2<c_m, t0> + c_m^T S c_m - w_m^T Mcap^-1 w_m
		  w_m = diag(a) (t0 - S c_m),  t0 = U^T D^-1 resid,  S = U^T D^-1 U.
		The (component-independent) logdet is pulled out of the logsumexp.
		With c = 0 this reduces exactly to ``_factor_gaussian_nll`` for any M
		and any pi (same per-dim scaling, so ``lambda_latent_mse`` transfers).
		When training, stashes detached responsibility sentinels in
		``_mdn_train_stats`` (read and cleared by ``forward_motion``).
		"""
		B, K, D = mu.shape
		kd = K * D
		resid = (target - mu).reshape(B, kd)
		lv = logvar.clamp(-8.0, 8.0).reshape(B, kd)
		dinv = torch.exp(-lv)                                  # (B, KD)
		a = torch.exp(log_a.clamp(-6.0, 4.0))                  # (B, r)
		u = self.latent_factors                                # (KD, r)
		ud = u.unsqueeze(0) * dinv.unsqueeze(-1)               # (B, KD, r) = D^-1 U
		s_mat = torch.einsum('dr,bds->brs', u, ud)             # (B, r, r) = U^T D^-1 U
		m_cap = a.unsqueeze(-1) * s_mat * a.unsqueeze(-2)
		m_cap = m_cap + torch.eye(
			m_cap.shape[-1], device=m_cap.device, dtype=m_cap.dtype)
		chol = torch.linalg.cholesky(m_cap)                    # ONE per sample (tied cov)
		t0 = torch.einsum('bd,bdr->br', resid, ud)             # (B, r) = U^T D^-1 resid
		q0 = (resid.pow(2) * dinv).sum(-1)                     # (B,)
		sc = torch.einsum('brs,bms->bmr', s_mat, c)            # (B, M, r) = S c_m
		quad_diag = (q0.unsqueeze(-1)
		             - 2.0 * (c * t0.unsqueeze(1)).sum(-1)
		             + (c * sc).sum(-1))                       # (B, M)
		w = a.unsqueeze(1) * (t0.unsqueeze(1) - sc)            # (B, M, r)
		w_rhs = w.transpose(1, 2)                              # (B, r, M): M RHS at once
		corr = (w_rhs * torch.cholesky_solve(w_rhs, chol)).sum(1)  # (B, M)
		quad = quad_diag - corr                                # (B, M) Mahalanobis
		logdet = lv.sum(-1) + 2.0 * torch.log(
			torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)    # (B,)
		ll = log_pi - 0.5 * quad                               # (B, M)
		nll = (0.5 * logdet - torch.logsumexp(ll, dim=-1)) / kd
		if self.training:
			with torch.no_grad():
				resp = torch.softmax(ll, dim=-1)               # posterior responsibilities
				usage = resp.mean(dim=0)                       # (M,)
				pi = log_pi.exp()
				self._mdn_train_stats = {
					'pi_ent': -(pi * log_pi).sum(-1).mean(),
					'use_min': usage.min(),
					'post_max': resp.max(dim=-1).values.mean(),
				}
		out = nll.mean()
		bal = float(self.hparams.latent_mdn_balance)
		if bal > 0.0:
			# KL(batch-mean posterior usage || uniform): rescue-only load
			# balancing on the mixture side (never touches mu; default 0).
			usage_live = torch.softmax(ll, dim=-1).mean(dim=0)
			m_comp = float(usage_live.shape[0])
			out = out + bal * (usage_live
			                   * (usage_live.clamp_min(1e-8) * m_comp).log()).sum()
		return out

	def _factor_sample(self, mu: torch.Tensor, logvar: torch.Tensor,
	                   log_a: torch.Tensor, temperature: float) -> torch.Tensor:
		"""One-shot structured draw ``z = mu + temp*(U(a*eps1) + sigma*eps2)``."""
		B, K, D = mu.shape
		a = torch.exp(log_a.clamp(-6.0, 4.0))                  # (B, r)
		eps1 = torch.randn_like(a)
		factor_part = torch.einsum('dr,br->bd', self.latent_factors, a * eps1)
		sigma = torch.exp(0.5 * logvar).reshape(B, K * D)
		diag_part = sigma * torch.randn_like(sigma)
		z = mu.reshape(B, K * D) + float(temperature) * (factor_part + diag_part)
		return z.reshape(B, K, D)

	def _mdn_sample(self, mu: torch.Tensor, logvar: torch.Tensor,
	                log_a: torch.Tensor, log_pi: torch.Tensor,
	                c: torch.Tensor, temperature: float) -> torch.Tensor:
		"""Mixture draw ``z = mu + temp*(U(c_m + a*eps1) + sigma*eps2)``.

		``m ~ Cat(pi(text))`` picks ONE coherent set-level mode per row (each
		MM-pass repeat row draws its own component). ``temp`` stays the same
		pure multiplier on the whole zero-mean perturbation around mu as in
		every other sampler branch (the mixture keeps ``sum_m pi_m c_m ~ 0``
		because mu owns the conditional mean), so the calibration-sweep
		infrastructure applies verbatim. pi is NOT tempered: the weights are
		a discrete choice, not a noise magnitude.
		"""
		B, K, D = mu.shape
		m_idx = torch.multinomial(log_pi.exp(), 1).squeeze(-1)          # (B,)
		c_sel = c.gather(
			1, m_idx.view(B, 1, 1).expand(-1, 1, c.shape[-1])).squeeze(1)  # (B, r)
		a = torch.exp(log_a.clamp(-6.0, 4.0))                  # (B, r)
		eps1 = torch.randn_like(a)
		struct = torch.einsum(
			'dr,br->bd', self.latent_factors, c_sel + a * eps1)  # center + factor noise
		sigma = torch.exp(0.5 * logvar).reshape(B, K * D)
		z = mu.reshape(B, K * D) + float(temperature) * (
			struct + sigma * torch.randn_like(sigma))
		return z.reshape(B, K, D)

	def _tokenize_t2m_prompts(self, texts: List[str]):
		"""Tokenize the (left-padded) T2M prompts ending in the <MOT> seed."""
		prompts = self._build_t2m_inputs(texts)
		return self.tokenizer(
			prompts,
			return_tensors='pt',
			padding=True,
			truncation=True,
			max_length=self.lm.max_length,
		).to(self.device)

	def _embed_latent_feedback(self, z_norm: torch.Tensor) -> torch.Tensor:
		"""Standardized latent value(s) -> GPT-2 input embedding(s)."""
		return self.latent_in_norm(self.latent_in_projector(z_norm))

	def _encode_text_ar_teacher_forced(
		self, texts: List[str], z_gt_norm: torch.Tensor,
	) -> torch.Tensor:
		"""Teacher-forced chain: ONE forward over ``[prompt <MOT> z_1 .. z_{K-1}]``.

		Returns the hiddens at the K prediction positions
		``[<MOT>, z_1, .., z_{K-1}]`` (B, K, H): position k-1 predicts slot k
		given the REAL previous latents (causal mask does the rest). With the
		left-padded prompt these are exactly the last K columns.
		"""
		K = int(self.hparams.k_motion_tokens)
		enc = self._tokenize_t2m_prompts(texts)
		input_ids, attention_mask = enc['input_ids'], enc['attention_mask']
		inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
		if K > 1:
			z_embeds = self._embed_latent_feedback(z_gt_norm[:, :K - 1])  # (B, K-1, H)
			inputs_embeds = torch.cat([inputs_embeds, z_embeds], dim=1)
			attention_mask = torch.cat(
				[attention_mask,
				 attention_mask.new_ones(attention_mask.shape[0], K - 1)], dim=1)
		out = self.language_model(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			output_hidden_states=True,
			use_cache=False,
			return_dict=True,
		)
		return out.hidden_states[-1][:, -K:, :]  # (B, K, H)

	@torch.no_grad()
	def _ar_rollout(self, texts: List[str], temperature: float):
		"""Free-running chain: sample slot k, feed it back, predict slot k+1.

		``temperature`` 0 feeds back the per-step mean mu_k (deterministic
		chain); > 0 samples ``z_k = mu_k + temp*sigma_k*eps``. Always uses the
		KV cache (K-1 single-token steps). Returns ``(z_norm, mu, logvar)``,
		each (B, K, D) -- callers must decode THIS z, not a fresh draw from
		(mu, logvar): mu_{k+1} is conditioned on the z_k drawn here.
		"""
		K = int(self.hparams.k_motion_tokens)
		enc = self._tokenize_t2m_prompts(texts)
		cur_attn = enc['attention_mask']
		out = self.language_model(
			input_ids=enc['input_ids'],
			attention_mask=cur_attn,
			output_hidden_states=True,
			use_cache=True,
			return_dict=True,
		)
		h = out.hidden_states[-1][:, -1:, :]  # <MOT> hidden
		past = out.past_key_values
		zs, mus, lvs = [], [], []
		for k in range(K):
			params = self.projector(self.projector_norm(h))  # (B, 1, 2D)
			mu_k, lv_k = params.chunk(2, dim=-1)
			lv_k = lv_k.clamp(-10.0, 10.0)
			if temperature > 0.0:
				std_k = torch.exp(0.5 * lv_k) * float(temperature)
				z_k = mu_k + std_k * torch.randn_like(std_k)
			else:
				z_k = mu_k
			mus.append(mu_k)
			lvs.append(lv_k)
			zs.append(z_k)
			if k < K - 1:
				cur_attn = torch.cat(
					[cur_attn, cur_attn.new_ones(cur_attn.shape[0], 1)], dim=1)
				step = self.language_model(
					inputs_embeds=self._embed_latent_feedback(z_k),
					attention_mask=cur_attn,
					past_key_values=past,
					output_hidden_states=True,
					use_cache=True,
					return_dict=True,
				)
				past = step.past_key_values
				h = step.hidden_states[-1][:, -1:, :]
		return (torch.cat(zs, dim=1), torch.cat(mus, dim=1), torch.cat(lvs, dim=1))

	def _build_t2m_inputs(self, texts: List[str]) -> List[str]:
		"""Per-sample T2M prompt, ending in the <MOT> seed that starts the rollout.

		The GNU phase uses the explicit instruction-style prompt; the default
		(plain T2M) phase keeps the original short prompt so existing models /
		checkpoints are byte-for-byte unaffected.
		"""
		if getattr(self, 'is_gnu', False):
			return [
				f'Please generate human motion based on the following textual '
				f'description: {t.strip()} {self.mot_token_str}' for t in texts
			]
		return _build_inputs(texts, self.mot_token_str)

	def _encode_text_ar(self, texts: List[str]) -> torch.Tensor:
		"""AR path: roll K hidden states autoregressively, optionally with KV cache.

		Returns
		-------
		hidden : (B, K, hidden_size)
		"""
		K = int(self.hparams.k_motion_tokens)
		device = self.device

		# 1) Tokenize prompts ending with <MOT>.
		prompts = self._build_t2m_inputs(texts)
		enc = self.tokenizer(
			prompts,
			return_tensors='pt',
			padding=True,
			truncation=True,
			max_length=self.lm.max_length,
		).to(device)

		input_ids = enc['input_ids']
		attention_mask = enc['attention_mask']

		use_cache = bool(self.hparams.use_kv_cache)

		# 2) First step: run full prompt; the <MOT> hidden is the last column
		# (the tokenizer used in this codebase pads on the left for decoder LMs).
		out = self.language_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,
			use_cache=use_cache,
			return_dict=True,
		)
		last_hidden_layer = out.hidden_states[-1]  # (B, L, H)
		h0 = last_hidden_layer[:, -1:, :]  # (B, 1, H) — <MOT> hidden
		past = out.past_key_values if use_cache else None

		hiddens = [h0]
		cur_attn = attention_mask
		prompt_embeds = None
		for _ in range(K - 1):
			next_input = self.feedback_norm(hiddens[-1])  # (B, 1, H)
			if use_cache:
				cur_attn = torch.cat(
					[cur_attn, torch.ones((cur_attn.shape[0], 1), dtype=cur_attn.dtype, device=device)],
					dim=1,
				)
				step = self.language_model(
					inputs_embeds=next_input,
					attention_mask=cur_attn,
					past_key_values=past,
					output_hidden_states=True,
					use_cache=True,
					return_dict=True,
				)
				past = step.past_key_values
				new_h = step.hidden_states[-1][:, -1:, :]
			else:
				# Recompute the full context plus all previously generated hidden
				# states. Mix token ids (prompt) with inputs_embeds (generated)
				# by converting the prompt to embeddings up-front.
				if prompt_embeds is None:
					prompt_embeds = self.language_model.get_input_embeddings()(input_ids)
				cat_h = torch.cat(hiddens, dim=1)  # (B, t, H)
				full_embeds = torch.cat([prompt_embeds, self.feedback_norm(cat_h)], dim=1)
				step_attn = torch.cat(
					[
						attention_mask,
						torch.ones((attention_mask.shape[0], cat_h.shape[1]), dtype=attention_mask.dtype, device=device),
					],
					dim=1,
				)
				step = self.language_model(
					inputs_embeds=full_embeds,
					attention_mask=step_attn,
					output_hidden_states=True,
					use_cache=False,
					return_dict=True,
				)
				new_h = step.hidden_states[-1][:, -1:, :]
			hiddens.append(new_h)

		return torch.cat(hiddens, dim=1)  # (B, K, H)

	# ----------------------------------------------------------------------
	# Loss computation
	# ----------------------------------------------------------------------
	@staticmethod
	def _lengths_to_mask(lengths, max_len: int, device):
		lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
		idx = torch.arange(max_len, device=device).unsqueeze(0)
		return idx < lengths_t.unsqueeze(1)

	@staticmethod
	def _latent_pairwise_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		"""Pairwise per-dim squared distance between two latent sets."""
		return torch.cdist(a.float(), b.float()).pow(2) / max(1, a.shape[-1])

	def _latent_chamfer_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		d = self._latent_pairwise_mse(pred, target)
		return 0.5 * (d.min(dim=2).values.mean() + d.min(dim=1).values.mean())

	def _latent_sinkhorn_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""Balanced entropic OT over K latent tokens in standardized space."""
		cost = self._latent_pairwise_mse(pred, target)
		eps = max(float(self.hparams.latent_sinkhorn_epsilon), 1e-6)
		iters = max(int(self.hparams.latent_sinkhorn_iters), 1)
		B, K, _ = cost.shape
		log_k = -cost / eps
		log_mu = cost.new_full((B, K), -np.log(float(K)))
		log_nu = cost.new_full((B, K), -np.log(float(K)))
		u = cost.new_zeros(B, K)
		v = cost.new_zeros(B, K)
		for _ in range(iters):
			u = log_mu - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
			v = log_nu - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)
		plan = torch.exp(log_k + u.unsqueeze(2) + v.unsqueeze(1))
		return (plan * cost).sum(dim=(1, 2)).mean()

	def _is_nll_mode(self) -> bool:
		mode = str(self.hparams.latent_loss_mode).lower().replace('-', '_')
		return mode in ('nll', 'chamfer_nll')

	@staticmethod
	def _gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor,
	                  target: torch.Tensor) -> torch.Tensor:
		"""Per-dim Gaussian NLL ``0.5*((t-mu)^2/sigma^2 + log sigma^2)``.

		The proper scoring rule that makes ``sigma`` the ML estimate of the TRUE
		per-text/per-slot/per-dim conditional spread of the GT latents around
		``mu`` — the calibration that MSE + variance-only KL structurally cannot
		provide (MSE has zero logvar gradient; the KL's optimum is the constant
		sigma=1 regardless of the residuals). Constant ``0.5*log(2*pi)`` dropped.
		At convergence (sigma^2 ≈ residual^2) the mu-gradient ``(mu-t)/sigma^2``
		matches the scale of the old MSE gradient, so ``lambda_latent_mse``
		transfers as the weight unchanged.
		"""
		lv = logvar.clamp(-8.0, 8.0)  # sigma in [~0.018, ~54]: NLL blow-up guard
		return 0.5 * ((target - mu).pow(2) * torch.exp(-lv) + lv).mean()

	def _latent_chamfer_nll(self, mu: torch.Tensor, logvar: torch.Tensor,
	                        target: torch.Tensor) -> torch.Tensor:
		"""Symmetric chamfer-matched Gaussian NLL over the K latent slots.

		Matching is computed on mu distances (no grad through the assignment,
		mirroring the plain chamfer anchor); the NLL is then evaluated on the
		matched pairs in both directions.
		"""
		B, K, D = mu.shape
		with torch.no_grad():
			d = self._latent_pairwise_mse(mu, target)  # (B, K_pred, K_tgt)
			j_star = d.argmin(dim=2)                   # pred slot -> nearest GT
			i_star = d.argmin(dim=1)                   # GT slot   -> nearest pred
		tgt1 = torch.gather(target, 1, j_star.unsqueeze(-1).expand(-1, -1, D))
		nll1 = self._gaussian_nll(mu, logvar, tgt1)
		mu2 = torch.gather(mu, 1, i_star.unsqueeze(-1).expand(-1, -1, D))
		lv2 = torch.gather(logvar, 1, i_star.unsqueeze(-1).expand(-1, -1, D))
		nll2 = self._gaussian_nll(mu2, lv2, target)
		return 0.5 * (nll1 + nll2)

	def _latent_anchor_loss(self, mu: torch.Tensor, target: torch.Tensor,
	                        logvar: Optional[torch.Tensor] = None,
	                        log_a: Optional[torch.Tensor] = None,
	                        log_pi: Optional[torch.Tensor] = None,
	                        c: Optional[torch.Tensor] = None):
		"""Return optimized latent anchor plus fixed/set diagnostics."""
		mode = str(self.hparams.latent_loss_mode).lower()
		mode = mode.replace('-', '_')
		fixed = F.mse_loss(mu, target)
		chamfer = self._latent_chamfer_loss(mu, target)

		if mode in ('fixed', 'mse', 'slot_mse'):
			return fixed, fixed, chamfer
		if mode == 'chamfer':
			return chamfer, fixed, chamfer
		if mode == 'sinkhorn':
			sinkhorn = self._latent_sinkhorn_loss(mu, target)
			return sinkhorn, fixed, sinkhorn
		if mode == 'fixed_chamfer':
			return fixed + float(self.hparams.lambda_latent_set) * chamfer, fixed, chamfer
		if mode == 'fixed_sinkhorn':
			sinkhorn = self._latent_sinkhorn_loss(mu, target)
			return fixed + float(self.hparams.lambda_latent_set) * sinkhorn, fixed, sinkhorn
		if mode in ('nll', 'chamfer_nll'):
			if self.latent_low_rank > 0 and log_a is not None:
				# DECOUPLED mean/covariance training (2026-07-03 night fix).
				# Joint ML of (mu, Sigma) lets the covariance absorb mu's own
				# prediction error: within 10 epochs factor_a_mean exploded
				# 0.15 -> 4.2 (a_max 13.7), the NLL mu-gradient Sigma^-1(mu-z)
				# lost its precision along the inflated factors, and mu's
				# fixed_mse STALLED (1.47 -> 1.37 vs 1.47 -> 1.0 for the
				# diagonal run) while eval R@1 collapsed to 0.13 (temp-1
				# samples swamped by factor noise fit to mu-error-dominated
				# residuals) -- the classic heteroscedastic-regression
				# variance-eats-gradient failure. Fix: mu learns under the
				# PROVEN diagonal NLL with detached precision; the covariance
				# side (logvar, a, U) does maximum likelihood on the DETACHED
				# residual. No back-door from Sigma to mu; as mu improves the
				# residual shrinks and a deflates toward the true conditional
				# spread (self-correcting, no clamps needed).
				tgt_mu = tgt_cov = target
				nll_mu = None
				if mode == 'chamfer_nll':
					# K-scaling regime (2026-07-04, for K >= 8): the encoder's
					# slot assignment is arbitrary per sample, so the FIXED-slot
					# residual is inflated by an assignment component that decode
					# (near-permutation-invariant over slots) never sees. A
					# covariance calibrated on it oversamples noise at eval.
					# Chamfer-match GT slots to the prediction's slot order (mu
					# distances, no grad -- mirrors _latent_chamfer_nll) and run
					# BOTH the decoupled mu anchor and the covariance ML on the
					# matched residual: sigma/a then estimate the decode-relevant
					# conditional spread, not the assignment artifact. The
					# factor NLL keeps the prediction's slot order for the
					# flattened K*D covariance, so U's semantics are stable.
					B, K, D = mu.shape
					with torch.no_grad():
						d = self._latent_pairwise_mse(mu, target)
						j_star = d.argmin(dim=2)   # pred slot -> nearest GT
						i_star = d.argmin(dim=1)   # GT slot   -> nearest pred
					tgt_cov = torch.gather(
						target, 1, j_star.unsqueeze(-1).expand(-1, -1, D))
					tgt_mu = tgt_cov
					mu2 = torch.gather(
						mu, 1, i_star.unsqueeze(-1).expand(-1, -1, D))
					lv2 = torch.gather(
						logvar, 1, i_star.unsqueeze(-1).expand(-1, -1, D))
					nll_mu = 0.5 * (
						self._gaussian_nll(mu, logvar.detach(), tgt_mu)
						+ self._gaussian_nll(mu2, lv2.detach(), target))
				if nll_mu is None:
					nll_mu = self._gaussian_nll(mu, logvar.detach(), target)
				if self.latent_mdn_components > 0 and log_pi is not None:
					# Mixture ML on the mu-detached residual: (pi, c, a, U,
					# sigma) estimate the multi-peak conditional residual
					# density; no back-door into mu (the exact contract of
					# the single-component factor NLL below).
					nll_cov = self._mdn_mixture_nll(
						mu.detach(), logvar, log_a, log_pi, c, tgt_cov)
				else:
					nll_cov = self._factor_gaussian_nll(
						mu.detach(), logvar, log_a, tgt_cov)
				return nll_mu + nll_cov, fixed, chamfer
			if mode == 'chamfer_nll':
				return self._latent_chamfer_nll(mu, logvar, target), fixed, chamfer
			return self._gaussian_nll(mu, logvar, target), fixed, chamfer
		raise ValueError(
			"latent_loss_mode must be one of: fixed, chamfer, sinkhorn, "
			"fixed_chamfer, fixed_sinkhorn, nll, chamfer_nll"
		)

	def forward_motion(self, batch):
		"""Run the full pipeline and return predicted motion + diagnostics."""
		motion = batch['motion']
		lengths = batch['length']
		texts = batch['text']

		B, T, _ = motion.shape
		device = motion.device

		# Frozen encoder GT latents (no grad through encoder).
		with torch.no_grad():
			latents_gt = self.alae_wrapper.encode_latents(motion)

		# Distribution over standardized latents. In AR mode the GT latents are
		# teacher-forced as chain context, so slot k's (mu, logvar) is the
		# conditional given the REAL z_{<k} and the per-slot NLL below sums to
		# the exact chain-rule likelihood of the GT latent set.
		latents_gt_norm = self._standardize_latents(latents_gt)
		log_a = log_pi = mdn_c = None
		if self.latent_low_rank > 0:
			mu, logvar, log_a, log_pi, mdn_c = self._compute_latent_dist_factor(texts)
		else:
			teacher_ctx = None
			if self.ar_latent_sampling:
				teacher_ctx = latents_gt_norm
				if bool(self.hparams.ar_context_sampling) and self.training:
					# Scheduled sampling (two-pass): perturb the chain context with
					# each slot's OWN predictive draw given the GT prefix, so the
					# graded pass learns to predict the true slot from an imperfect
					# context (see the ar_context_sampling doc in __init__). The
					# NLL target below stays the GT latents.
					with torch.no_grad():
						h0 = self._encode_text_ar_teacher_forced(texts, latents_gt_norm)
						p0 = self.projector(self.projector_norm(h0))
						mu0, lv0 = p0.chunk(2, dim=-1)
						lv0 = lv0.clamp(-10.0, 10.0)
						teacher_ctx = mu0 + torch.exp(0.5 * lv0) * torch.randn_like(mu0)
			mu, logvar = self._compute_latent_dist(texts, teacher_latents=teacher_ctx)
		logvar = logvar.clamp(-10.0, 10.0)                       # numerical guardrail
		if self._is_nll_mode():
			# NLL recipe (2026-07-03): the per-sample losses (rec/ric/percept)
			# supervise the conditional MEAN — decode mu, never a sampled z.
			# Decoding z = mu + sigma*eps against the FIXED paired target is what
			# taught the pipeline latent-noise invariance (with the KL pinning
			# sigma at 1, training decoded mu + marginal-scale noise onto one GT;
			# the decoder then ignored sigma-scale perturbations at eval, so no
			# sampling temperature could restore within-text variance). sigma is
			# trained ONLY by the NLL calibration term; sampling happens at eval.
			z_norm = mu
		else:
			# Legacy recipe: reconstruct from a sampled z so the sigma-ball is
			# trained to be decodable.
			z_norm = self._reparameterize(mu, logvar)            # sample (train)
		latents_pred = self._destandardize_latents(z_norm)       # raw -> decoder
		latent_mse, latent_fixed_mse, latent_set = self._latent_anchor_loss(
			mu, latents_gt_norm, logvar=logvar, log_a=log_a,
			log_pi=log_pi, c=mdn_c,
		)
		kl = self._kl_variance(logvar)  # variance-only; diagnostic-only in NLL mode
		sigma = torch.exp(0.5 * logvar)
		sigma_mean = sigma.mean().detach()
		# Cross-sample spread of the per-sample mean sigma. The dead-sigma
		# failure mode (KL-pinned) shows mean~1 and bstd~0; a live calibrated
		# sigma must vary with the text, i.e. bstd clearly > 0.
		sigma_bstd = (sigma.mean(dim=(1, 2)).std().detach()
		              if sigma.shape[0] > 1 else sigma.new_zeros(()))
		# Factor-head diagnostics: the amplitudes must GROW from the e^-2 start
		# (structure being learned) and vary across texts; a_mean stuck at 0.14
		# means the factors never engaged.
		factor_a_mean = factor_a_max = None
		if log_a is not None:
			a_amp = torch.exp(log_a.clamp(-6.0, 4.0)).detach()
			factor_a_mean, factor_a_max = a_amp.mean(), a_amp.max()
		# MDN mixture sentinels (None when off). pi_ent starts at ln M and
		# should DECLINE slowly (text-conditional weights); use_min -> 0 =
		# starvation; post_max rising from 1/M = specialization; c_rms must
		# GROW from the init (~offset_init) then roll over as fixed_mse
		# grinds down (the factor_a rise-then-rollover shape); off_bias
		# (RMS of sum_m pi_m c_m) must stay << c_rms (mixture centered on mu).
		mdn_pi_ent = mdn_use_min = mdn_post_max = None
		mdn_c_rms = mdn_off_bias = None
		if mdn_c is not None:
			stats = self._mdn_train_stats
			self._mdn_train_stats = None
			if stats is not None:
				mdn_pi_ent = stats['pi_ent']
				mdn_use_min = stats['use_min']
				mdn_post_max = stats['post_max']
			with torch.no_grad():
				mdn_c_rms = mdn_c.pow(2).mean().sqrt()
				mdn_off_bias = (log_pi.exp().unsqueeze(-1)
				                * mdn_c).sum(1).pow(2).mean().sqrt()

		x_hat = self.alae_wrapper.decode_to_motion(latents_pred, target_len=T)

		mask = self._lengths_to_mask(lengths, T, device)
		lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)

		self._ensure_t2m_encoders()
		alae_losses = self.alae_wrapper.compute_loss(
			motion,
			x_hat,
			latents_pred,
			mask=mask,
			lengths=lengths_t,
			t2m_move_enc=self._t2m_move_enc,
			t2m_motion_enc=self._t2m_motion_enc,
			percept_fn=self._percept_fn(),
			lambda_rec=self.hparams.lambda_rec,
			lambda_ric=self.hparams.lambda_ric,
			lambda_percept=self.hparams.lambda_percept,
			lambda_latent_l2=self.hparams.lambda_latent_l2,
			ric_slice=self._ric_slice,
			use_percept_cosine=self.hparams.use_percept_cosine,
			lambda_percept_cosine=self.hparams.lambda_percept_cosine,
		)

		total = alae_losses['loss'] + float(self.hparams.lambda_latent_mse) * latent_mse
		beta = self._kl_beta()
		if beta != 0.0:
			total = total + beta * kl

		return {
			'loss': total,
			'loss_rec': alae_losses['loss_rec'],
			'loss_ric': alae_losses['loss_ric'],
			'loss_percept': alae_losses['loss_percept'],
			'loss_percept_cos': alae_losses['loss_percept_cos'],
			'loss_latent_l2': alae_losses['loss_latent_l2'],
			'loss_latent_mse': latent_mse,
			'loss_latent_fixed_mse': latent_fixed_mse,
			'loss_latent_set': latent_set,
			'loss_kl': kl if torch.is_tensor(kl) else latents_pred.new_tensor(kl),
			'kl_beta': latents_pred.new_tensor(float(beta)),
			'sigma_mean': sigma_mean,
			'sigma_bstd': sigma_bstd,
			'factor_a_mean': factor_a_mean,
			'factor_a_max': factor_a_max,
			'mdn_pi_ent': mdn_pi_ent,
			'mdn_use_min': mdn_use_min,
			'mdn_post_max': mdn_post_max,
			'mdn_c_rms': mdn_c_rms,
			'mdn_off_bias': mdn_off_bias,
			'x_hat': x_hat,
			'latents_pred': latents_pred,
			'latents_gt': latents_gt,
		}

	# ----------------------------------------------------------------------
	# GNU: motion-to-text (understanding) branch
	# ----------------------------------------------------------------------
	def _m2t_input_embeds(self, motion: torch.Tensor):
		"""Build M2T prompt embeddings ``[prefix text, <MOT>, K motion tokens]``.

		The K motion tokens are the frozen ALAE-encoder latents pushed through the
		trainable ``motion_in_projector``, so gradients reach the projector but not
		the (frozen) encoder. Returns ``(prompt_embeds (B, S, H), attn (B, S))``;
		the prompt length S is identical for every sample, so no padding is needed.
		"""
		B = motion.shape[0]
		device = motion.device
		with torch.no_grad():
			latents_gt = self.alae_wrapper.encode_latents(motion)  # (B, K, D)
		noise_std = float(self.hparams.m2t_latent_noise_std)
		if self.training and noise_std > 0.0:
			noise = torch.randn_like(latents_gt)
			std = getattr(self, 'latent_std', None)  # (K, D); variational only
			if std is not None:
				noise = noise * std
			latents_gt = latents_gt + noise_std * noise
		motion_embeds = self.motion_in_projector(latents_gt)        # (B, K, H)

		wte = self.language_model.get_input_embeddings()
		prefix_ids = self._m2t_prefix_ids.to(device).unsqueeze(0).expand(B, -1)  # (B, P)
		prefix_embeds = wte(prefix_ids)                                          # (B, P, H)
		mot_ids = torch.full((B, 1), self.mot_token_id, dtype=torch.long, device=device)
		mot_embeds = wte(mot_ids)                                                # (B, 1, H)

		prompt_embeds = torch.cat([prefix_embeds, mot_embeds, motion_embeds], dim=1)
		attn = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
		return prompt_embeds, attn

	def forward_m2t(self, batch) -> torch.Tensor:
		"""Teacher-forced motion->text cross-entropy loss (the U in GNU)."""
		motion = batch['motion']
		texts = batch['text']
		device = motion.device

		prompt_embeds, prompt_attn = self._m2t_input_embeds(motion)  # (B, S, H)
		B, S, _ = prompt_embeds.shape

		# Tokenize target captions with a trailing EOS, RIGHT-padded: teacher
		# forcing needs the target contiguous right after the motion context, so
		# the global left-padding (used for generation) is flipped here.
		wte = self.language_model.get_input_embeddings()
		prev_side = self.tokenizer.padding_side
		self.tokenizer.padding_side = 'right'
		try:
			tgt = self.tokenizer(
				[t.strip() + self.tokenizer.eos_token for t in texts],
				return_tensors='pt', padding=True, truncation=True,
				max_length=int(self.hparams.m2t_max_new_tokens),
				add_special_tokens=False,
			)
		finally:
			self.tokenizer.padding_side = prev_side
		tgt_ids = tgt['input_ids'].to(device)        # (B, T)
		tgt_attn = tgt['attention_mask'].to(device)  # (B, T)
		tgt_embeds = wte(tgt_ids)                     # (B, T, H)

		inputs_embeds = torch.cat([prompt_embeds, tgt_embeds], dim=1)
		attn = torch.cat([prompt_attn, tgt_attn], dim=1)
		# Loss only on the target tokens; -100 on the prompt + padded positions.
		ignore = torch.full((B, S), -100, dtype=torch.long, device=device)
		tgt_labels = tgt_ids.masked_fill(tgt_attn == 0, -100)
		labels = torch.cat([ignore, tgt_labels], dim=1)

		# Forward WITHOUT labels= so we can compute the loss ourselves and apply
		# label smoothing. With m2t_label_smoothing=0 this is numerically identical
		# to HF's built-in ForCausalLMLoss (same causal shift + ignore_index).
		out = self.language_model(
			inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True,
		)
		# Causal shift: position t predicts token t+1.
		shift_logits = out.logits[:, :-1, :].contiguous()
		shift_labels = labels[:, 1:].contiguous()
		return F.cross_entropy(
			shift_logits.view(-1, shift_logits.size(-1)),
			shift_labels.view(-1),
			ignore_index=-100,
			label_smoothing=float(self.hparams.m2t_label_smoothing),
		)

	@torch.no_grad()
	def val_m2t_forward(self, batch) -> List[str]:
		"""Greedy-generate one caption per motion for M2T evaluation."""
		prompt_embeds, prompt_attn = self._m2t_input_embeds(batch['motion'])
		gen = self.language_model.generate(
			inputs_embeds=prompt_embeds,
			attention_mask=prompt_attn,
			max_new_tokens=int(self.hparams.m2t_max_new_tokens),
			do_sample=False,
			num_beams=int(self.hparams.m2t_num_beams),
			pad_token_id=self.tokenizer.eos_token_id,
		)
		# With inputs_embeds (no input_ids), HF returns only the new tokens.
		texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
		return [t.strip() for t in texts]

	def collect_m2t_records(self, enable: bool = True):
		"""Enable (and reset) or disable per-sample M2T result capture.

		When enabled, each M2T eval pass appends ``{'fname', 'gt_text',
		'pred_text'}`` rows to ``self._m2t_records`` so local_eval.py can dump
		them to m2t_results.csv. Resetting on enable keeps a single eval pass
		from accumulating across replications.
		"""
		self._m2t_records = [] if enable else None

	def _update_m2t_metrics(self, batch):
		if not hasattr(self.metrics, 'M2TMetrics'):
			return
		pred_texts = self.val_m2t_forward(batch)
		feats_ref = self.datamodule.renorm4t2m(batch['motion'])
		self.metrics.M2TMetrics.update(
			feats_ref=feats_ref,
			pred_texts=pred_texts,
			gt_texts=batch['all_captions'],
			lengths=batch['length'],
			word_embs=batch.get('word_embs'),
			pos_ohot=batch.get('pos_ohot'),
			text_lengths=batch.get('text_len'),
		)
		# Optional capture for the m2t_results.csv dump (local_eval.py). The GT
		# is one-to-many (multiple reference captions), so all references are
		# joined; the prediction is the single greedy/beam caption per motion.
		if self._m2t_records is not None:
			fnames = batch.get('fname')
			gt_caps = batch['all_captions']
			for i, pred in enumerate(pred_texts):
				refs = gt_caps[i]
				gt_text = ' | '.join(str(r) for r in refs) if isinstance(
					refs, (list, tuple)) else str(refs)
				fname = (str(fnames[i]) if fnames is not None
				         else f'sample_{len(self._m2t_records)}')
				self._m2t_records.append({
					'fname': fname,
					'gt_text': gt_text,
					'pred_text': pred,
				})

	# ----------------------------------------------------------------------
	# Lightning steps
	# ----------------------------------------------------------------------
	def allsplit_step(self, split: str, batch, batch_idx):
		batch_size = len(batch['text'])

		if split == 'train':
			out = self.forward_motion(batch)
			if self.is_gnu:
				# Understanding branch: fold motion->text loss into the same step.
				m2t_loss = self.forward_m2t(batch)
				# Logged as the raw CE (comparable across runs); the optimized
				# term is the flooded version below. Flooding working == the raw
				# CE flatlining at ~m2t_loss_floor instead of descending.
				out['loss_m2t'] = m2t_loss
				floor = float(self.hparams.m2t_loss_floor)
				if floor > 0.0:
					m2t_loss = (m2t_loss - floor).abs() + floor
				out['loss'] = out['loss'] + float(self.hparams.lambda_m2t) * m2t_loss
			# Per-step log for the progress bar (total loss only).
			self.log(
				'train/loss', out['loss'].detach(),
				on_step=True, on_epoch=False, prog_bar=True,
				logger=True, batch_size=batch_size,
			)
			# Epoch-averaged logs. Use clean keys (no _step/_epoch suffix) so the
			# shared progressLogger / LossCSVLogger / TensorBoard all see one
			# canonical name per metric. 'total/train' matches the existing
			# metric_monitor 'loss_total' alias in motGPT.callback.
			epoch_log_keys = {
				'loss': 'total/train',
				'loss_rec': 'train/loss_rec',
				'loss_ric': 'train/loss_ric',
				'loss_percept': 'train/loss_percept',
				'loss_percept_cos': 'train/loss_percept_cos',
				'loss_latent_l2': 'train/loss_latent_l2',
				'loss_latent_mse': 'train/loss_latent_mse',
				'loss_latent_fixed_mse': 'train/loss_latent_fixed_mse',
				'loss_latent_set': 'train/loss_latent_set',
				'loss_kl': 'train/loss_kl',
				'kl_beta': 'train/kl_beta',
				'sigma_mean': 'train/sigma_mean',
				'sigma_bstd': 'train/sigma_bstd',
				'factor_a_mean': 'train/factor_a_mean',
				'factor_a_max': 'train/factor_a_max',
				'mdn_pi_ent': 'train/mdn_pi_ent',
				'mdn_use_min': 'train/mdn_use_min',
				'mdn_post_max': 'train/mdn_post_max',
				'mdn_c_rms': 'train/mdn_c_rms',
				'mdn_off_bias': 'train/mdn_off_bias',
				'loss_m2t': 'train/loss_m2t',
			}
			for key, log_name in epoch_log_keys.items():
				val = out.get(key)
				if torch.is_tensor(val):
					self.log(
						log_name, val.detach(),
						on_step=False, on_epoch=True, prog_bar=False,
						logger=True, sync_dist=True, batch_size=batch_size,
					)
			return out['loss']

		if split in ('val', 'test'):
			# Route-A lever / diagnostic: optionally SAMPLE on the FID/R pass
			# (z = mu + temp*sigma*eps) instead of decoding the deterministic
			# mean mu. Decoding mu gives one motion per text -> Sigma_gen misses
			# its within-text component -> FID covariance mismatch (invisible to
			# Diversity, which is between-text only). Sampling restores it.
			# METRIC.FID_SAMPLE_TEMP=0 (default) -> mu; the mm pass is untouched.
			fid_temp = float(self.hparams.cfg.METRIC.get('FID_SAMPLE_TEMP', 0.0))
			if fid_temp > 0.0 and not self.trainer.datamodule.is_mm:
				rs_set = self.val_t2m_forward(
					batch, sample=True, temperature=fid_temp)
			else:
				rs_set = self.val_t2m_forward(batch)
			self._update_t2m_metrics(batch, rs_set)
			# GNU understanding branch: also accumulate M2T metrics on the normal
			# (non-mm) pass. mm mode repeats prompts for MultiModality only and has
			# no M2T notion, so skip it there.
			if self.is_gnu and not self.trainer.datamodule.is_mm:
				self._update_m2t_metrics(batch)
			return None

		return None

	# ----------------------------------------------------------------------
	# Metrics logging (GNU: TM2TMetrics + M2TMetrics, namespaced)
	# ----------------------------------------------------------------------
	def metrics_log_dict(self):
		"""Compute every configured metric group for logging.

		In the GNU phase both ``TM2TMetrics`` (T2M) and ``M2TMetrics`` (M2T) are
		active and share key names (``Matching_score`` / ``R_precision_top_k``).
		The base implementation would let them clobber each other in the flat
		``Metrics/{key}`` namespace, so the M2T group is prefixed with ``M2T_``
		(the NLG keys ``bleu_*`` / ``ROUGE_L`` / ``CIDEr`` / ``Bert_F1`` are unique
		and surface as ``M2T_bleu_1`` etc.). Non-GNU defers to the base impl.
		"""
		if not getattr(self, 'is_gnu', False):
			return super().metrics_log_dict()

		if self.trainer.datamodule.is_mm and 'TM2TMetrics' in self.hparams.metrics_dict:
			metrics_dicts = ['MMMetrics']
		else:
			metrics_dicts = [m for m in self.hparams.metrics_dict if m != 'MMMetrics']
		metrics_dicts = [m for m in metrics_dicts if hasattr(self.metrics, m)]

		log = {}
		for metric in metrics_dicts:
			if metric == 'M2TMetrics':
				# BERTScore reloads a large model onto the GPU and is the slow /
				# hang-prone step. Run it only at final test by default; set
				# METRIC.M2T_BERT_ON_VAL to also compute it on the per-epoch val
				# pass. (BLEU/ROUGE/CIDEr + R-precision always run.)
				bert_on_val = bool(self.hparams.cfg.METRIC.get('M2T_BERT_ON_VAL', False))
				self.metrics.M2TMetrics.skip_bert_score = (
					not self.trainer.testing and not bert_on_val)
			result = getattr(self.metrics, metric).compute(
				sanity_flag=self.trainer.sanity_checking)
			prefix = 'M2T_' if metric == 'M2TMetrics' else ''
			for key, value in result.items():
				log[f"Metrics/{prefix}{key}"] = (
					value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
				)
		return log

	# ----------------------------------------------------------------------
	# Variational latent head helpers (standardized latent space)
	# ----------------------------------------------------------------------
	def _standardize_latents(self, z: torch.Tensor) -> torch.Tensor:
		return (z - self.latent_mean) / self.latent_std

	def _destandardize_latents(self, z_norm: torch.Tensor) -> torch.Tensor:
		return z_norm * self.latent_std + self.latent_mean

	@staticmethod
	def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		return mu + std * torch.randn_like(std)

	def _kl_variance(self, logvar: torch.Tensor) -> torch.Tensor:
		"""Sum-over-(K,D), mean-over-batch *variance-only* KL regularizer.

		This is the variance part of KL( N(mu, sigma^2) || N(*, 1) ):
		``0.5 * (sigma^2 - 1 - log sigma^2)``, which depends ONLY on sigma. We
		deliberately drop the full N(0,I) KL's ``mu^2`` term: the mean is already
		supervised by MSE(mu, standardized GT), and penalizing mu toward 0 fights
		that anchor — it pins mu near 0 (especially as beta warms up), stalling
		the latent regression and keeping lat_mse high. With this term the mean is
		free to fit GT while sigma is still pulled toward 1 (the -log sigma^2 term
		blows up as sigma->0, preventing variance collapse -> preserves MM).
		``kl_free_bits`` (per-dim nats floor) lets sigma sit nearer 1 cheaply.
		"""
		kl = 0.5 * (torch.exp(logvar) - 1.0 - logvar)  # (B, K, D), sigma-only
		free_bits = float(self.hparams.kl_free_bits)
		if free_bits > 0.0:
			kl = kl.clamp_min(free_bits)
		return kl.sum(dim=(-2, -1)).mean()

	def _kl_beta(self) -> float:
		"""Linear warmup of the KL weight from 0 to ``lambda_kl`` over epochs."""
		beta_max = float(self.hparams.lambda_kl)
		warm = int(self.hparams.kl_warmup_epochs)
		if warm <= 0:
			return beta_max
		try:
			epoch = int(self.current_epoch)
		except (RuntimeError, AttributeError):
			epoch = 0
		return beta_max * min(max(epoch / float(warm), 0.0), 1.0)

	# ----------------------------------------------------------------------
	# Validation / TM2T
	# ----------------------------------------------------------------------
	@torch.no_grad()
	def val_t2m_forward(self, batch, sample: bool = False,
	                    temperature: Optional[float] = None):
		motion = batch['motion']
		texts = batch['text']
		lengths = batch['length']

		if self.trainer.datamodule.is_mm:
			repeats = self.hparams.cfg.METRIC.MM_NUM_REPEATS
			texts = list(texts) * repeats
			motion = motion.repeat_interleave(repeats, dim=0)
			lengths = list(lengths) * repeats

		B, T, _ = motion.shape

		# ``sample`` controls whether the FID / R-precision / MMDist pass uses
		# a draw ``z = mu + temp*sigma*eps`` or the deterministic mean ``mu``.
		# Default is mu (``METRIC.FID_SAMPLE_TEMP=0``). MultiModality samples
		# independently via compute_multimodality and therefore stays non-zero.
		temp = float(self.hparams.eval_sample_temperature
		             if temperature is None else temperature)
		if self.ar_latent_sampling:
			# Free-running chain. The decoded z MUST be the rollout's own draws
			# (mu_{k+1} is conditioned on the z_k sampled inside); temp 0 feeds
			# back the per-step means (deterministic chain).
			z_norm, mu, logvar = self._ar_rollout(
				texts, temp if sample else 0.0)
		elif self.latent_low_rank > 0:
			# Single-shot structured draw from the low-rank + diagonal
			# covariance (factor part carries the correlation structure).
			mu, logvar, log_a, log_pi, mdn_c = self._compute_latent_dist_factor(texts)
			logvar = logvar.clamp(-10.0, 10.0)
			# Diagnostic knob (eval-only, default off): scale ONLY the factor
			# part of the draw. 0 -> diagonal-only sampling (isolates the
			# factor contribution to FID/R); 1 -> normal. With the MDN head on
			# it scales every component's WITHIN-mode factor noise (the
			# centers are handled by FID_MDN_SCALE below).
			fac_scale = float(self.hparams.cfg.METRIC.get('FID_FACTOR_SCALE', 1.0))
			if fac_scale != 1.0:
				log_a = log_a + float(np.log(max(fac_scale, 1e-8)))
			if self.latent_mdn_components > 0 and mdn_c is not None:
				# Eval-only isolation knob: scale the component centers.
				# 0 -> centers off, the draw reduces exactly to
				# _factor_sample -- the one-flag A/B isolating the mixture's
				# FID/R contribution (the FID_FACTOR_SCALE precedent).
				mdn_scale = float(self.hparams.cfg.METRIC.get('FID_MDN_SCALE', 1.0))
				if mdn_scale != 1.0:
					mdn_c = mdn_c * mdn_scale
				if sample and temp > 0.0:
					z_norm = self._mdn_sample(
						mu, logvar, log_a, log_pi, mdn_c, temp)
				else:
					z_norm = mu
			elif sample and temp > 0.0:
				z_norm = self._factor_sample(mu, logvar, log_a, temp)
			else:
				z_norm = mu
		else:
			mu, logvar = self._compute_latent_dist(texts)
			logvar = logvar.clamp(-10.0, 10.0)
			if sample and temp > 0.0:
				std = torch.exp(0.5 * logvar) * temp
				z_norm = mu + std * torch.randn_like(std)
			else:
				z_norm = mu
		# ORACLE diagnostics (zero-training, NEVER report numbers; validated
		# 2026-07-03 on the fixed-length K4 ckpt):
		# - FID_ORACLE_S = s: decode z = mu + s*(z_gt - mu). s=1 reproduces
		#   Stage-1 recon (FID 1.28); the FID(s) curve upper-bounds any
		#   conditional sampler (24.3 / 19.6 / 9.9 / 2.8 / 1.28 at s=0..1).
		# - FID_ORACLE_SHUFFLE: batch-permuted residuals = a text-independent
		#   residual sampler (FID 19.3 but R@1 0.48 -> residuals are
		#   text-conditional). Gated off the MM pass (``not sample``).
		oracle_s = float(self.hparams.cfg.METRIC.get('FID_ORACLE_S', 0.0))
		oracle_shuffle = bool(self.hparams.cfg.METRIC.get('FID_ORACLE_SHUFFLE', False))
		if (oracle_s > 0.0 or oracle_shuffle) and not sample:
			z_gt_norm = self._standardize_latents(
				self.alae_wrapper.encode_latents(motion))
			if oracle_shuffle:
				perm = torch.randperm(z_gt_norm.shape[0], device=z_gt_norm.device)
				z_norm = mu + (z_gt_norm - mu)[perm]
			else:
				z_norm = mu + oracle_s * (z_gt_norm - mu)
		latents_pred = self._destandardize_latents(z_norm)

		x_hat = self.alae_wrapper.decode_to_motion(latents_pred, target_len=T)

		min_len = [max(1, int(length)) for length in lengths]
		joints_ref = self.feats2joints(motion)
		joints_rst = self.feats2joints(x_hat)

		feats_ref = self.datamodule.renorm4t2m(motion)
		feats_rst = self.datamodule.renorm4t2m(x_hat)

		return {
			'm_ref': feats_ref,
			'm_rst': feats_rst,
			'joints_ref': joints_ref,
			'joints_rst': joints_rst,
			'length': min_len,
		}

	def _update_t2m_metrics(self, batch, rs_set):
		if self.trainer.datamodule.is_mm:
			metrics_dicts = ['MMMetrics']
		else:
			metrics_dicts = [m for m in self.hparams.metrics_dict if m != 'MMMetrics']
		metrics_dicts = [metric for metric in metrics_dicts if hasattr(self.metrics, metric)]

		lengths = batch['length']
		for metric in metrics_dicts:
			if metric == 'TM2TMetrics' and self._is_snapmogen:
				# SnapMoGen evaluator encodes raw caption strings (T5 inside);
				# there are no GloVe word_embs/pos_ohot for this dataset.
				texts = batch['text']
				if self.trainer.datamodule.is_mm:
					repeats = self.hparams.cfg.METRIC.MM_NUM_REPEATS
					texts = [t for t in texts for _ in range(repeats)]
				getattr(self.metrics, metric).update(
					feats_ref=rs_set['m_ref'],
					feats_rst=rs_set['m_rst'],
					lengths_ref=lengths,
					lengths_rst=rs_set['length'],
					texts=texts,
				)
			elif metric == 'TM2TMetrics' and 'word_embs' in batch:
				word_embs = batch['word_embs']
				pos_ohot = batch['pos_ohot']
				text_lengths = batch['text_len']
				if self.trainer.datamodule.is_mm:
					repeats = self.hparams.cfg.METRIC.MM_NUM_REPEATS
					word_embs = word_embs.repeat_interleave(repeats, dim=0)
					pos_ohot = pos_ohot.repeat_interleave(repeats, dim=0)
					text_lengths = text_lengths.repeat_interleave(repeats, dim=0)
				getattr(self.metrics, metric).update(
					feats_ref=rs_set['m_ref'],
					feats_rst=rs_set['m_rst'],
					lengths_ref=lengths,
					lengths_rst=rs_set['length'],
					word_embs=word_embs,
					pos_ohot=pos_ohot,
					text_lengths=text_lengths,
				)
			elif metric == 'MMMetrics':
				getattr(self.metrics, metric).update(
					feats_rst=rs_set['m_rst'],
					lengths_rst=rs_set['length'],
				)

	# ----------------------------------------------------------------------
	# Per-epoch MultiModality monitor (validation) -> reported via TM2TMetrics
	# ----------------------------------------------------------------------
	# The normal val loop only feeds TM2TMetrics (FID / R-precision / MMDist);
	# MultiModality needs *repeats of the same prompt*, which that loop never
	# produces. When METRIC.MM_ON_VAL is set we run a small extra pass at
		# validation-epoch end (a fixed val subset, MM_VAL_NUM_REPEATS samples each
		# through the variational sampler),
	# embed with TM2T's own motion encoder, and stash the value into TM2TMetrics
	# so it is reported in the SAME metrics group as FID / R-precision. Off by
	# default; never touches the FID/R-precision accumulation.
	def on_validation_epoch_end(self):
		cfg = self.hparams.cfg
		if (not self.trainer.sanity_checking
				and bool(cfg.METRIC.get('MM_ON_VAL', False))
				and hasattr(self.metrics, 'TM2TMetrics')):
			try:
				mm = self._compute_val_multimodality()
				if mm is not None:
					# Picked up by TM2TMetrics.compute() inside super() below.
					self.metrics.TM2TMetrics._extra_multimodality = mm
			except Exception as e:  # never let the monitor crash training
				if self.global_rank == 0:
					print(f'[MM-val] skipped (epoch {self.current_epoch}): {e}')
		super().on_validation_epoch_end()

	def on_test_epoch_end(self):
		"""Same as the val monitor but for the test split, so MultiModality shows
		up in the test metrics table (and local_eval's aggregate) right alongside
		FID / R-precision. Uses the full MM_NUM_SAMPLES / MM_NUM_REPEATS for a
		report-grade number."""
		cfg = self.hparams.cfg
		if (not self.trainer.sanity_checking
				and hasattr(self.metrics, 'TM2TMetrics')):
			try:
				mm = self.compute_multimodality(
					self.datamodule.test_dataset,
					int(cfg.METRIC.MM_NUM_SAMPLES),
					int(cfg.METRIC.MM_NUM_REPEATS),
				)
				if mm is not None:
					# Picked up by TM2TMetrics.compute() inside super() below.
					self.metrics.TM2TMetrics._extra_multimodality = mm
			except Exception as e:
				if self.global_rank == 0:
					print(f'[MM-test] skipped: {e}')
		super().on_test_epoch_end()

	def _compute_val_multimodality(self):
		"""Thin wrapper: MultiModality monitor over a fixed val subset."""
		cfg = self.hparams.cfg
		return self.compute_multimodality(
			self.datamodule.val_dataset,
			int(cfg.METRIC.get('MM_VAL_NUM_SAMPLES', 32)),
			int(cfg.METRIC.get('MM_VAL_NUM_REPEATS', cfg.METRIC.MM_NUM_REPEATS)),
		)

	@torch.no_grad()
	def compute_multimodality(self, dataset, n_samples, repeats):
		"""Capped intra-prompt MultiModality over a fixed subset of ``dataset``.

		The single source of truth for MM in this codebase: generate ``repeats``
		motions per prompt with the (variational) sampler, embed each with
		TM2TMetrics' own motion encoder (the space FID / R-precision use), and
		return the mean pairwise distance. Used both by the per-epoch validation
		monitor and by local_eval.py (test split), so train- and test-time MM are
		computed identically. Returns a float, or None if no data.
		"""
		from torch.utils.data import DataLoader, Subset
		from motGPT.data.utils import humanml3d_collate
		from motGPT.metrics.utils import calculate_multimodality_np

		cfg = self.hparams.cfg
		mm_num_times = int(cfg.METRIC.MM_NUM_TIMES)
		repeats = int(repeats)
		# calculate_multimodality_np requires repeats strictly > mm_num_times.
		if repeats <= mm_num_times:
			repeats = mm_num_times + 1

		total = len(dataset)
		if total < 1:
			return None
		n = min(int(n_samples), total)
		# Fixed subset (seeded) so the value is comparable across epochs / reps.
		g = torch.Generator().manual_seed(int(cfg.get('SEED_VALUE', 0)))
		idxs = torch.randperm(total, generator=g)[:n].tolist()
		loader = DataLoader(
			Subset(dataset, idxs), batch_size=min(n, 64), shuffle=False,
			num_workers=0, collate_fn=humanml3d_collate,
		)

		tm2t = self.metrics.TM2TMetrics
		per_prompt_embs = []  # each (repeats, emb_dim)
		for batch in loader:
			motions = batch['motion']
			texts = batch['text']
			lengths = batch['length']
			for j in range(len(texts)):
				# One prompt -> repeats copies; the variational sampler draws
				# independent eps per row, so val_t2m_forward yields a diverse set
				# even though is_mm is False (no datamodule state is touched).
				motion_j = motions[j:j + 1].float().to(self.device)  # (1, T, 263)
				rep_batch = {
					'text': [texts[j]] * repeats,
					'motion': motion_j.repeat(repeats, 1, 1),
					'length': [int(lengths[j])] * repeats,
				}
				rs = self.val_t2m_forward(rep_batch, sample=True)
				emb = tm2t.get_motion_embeddings(rs['m_rst'], rs['length'])  # (R, E)
				per_prompt_embs.append(emb.detach().cpu())
		if not per_prompt_embs:
			return None
		# (num_prompts, repeats, emb_dim) -> mean pairwise distance over repeats.
		activation = torch.stack(per_prompt_embs, dim=0).numpy()
		return float(calculate_multimodality_np(activation, mm_num_times))
