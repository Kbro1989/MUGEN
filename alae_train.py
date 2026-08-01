import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

import matplotlib

project_root = '../'
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from adaptive_length_auto_encoder import AdaptiveLengthAutoEncoder
from dataloader.humanml3d.humanml3d_263_dataset_mgpt import (
	HumanML3DVarLenDataset,
	varlen_collate,
)
from dataloader.snapmogen.snapmogen_296_dataset import SnapMoGenVarLenDataset
from utils.ae_snapmogen_eval import (
	SnapMoGenTextMotionEvalDataset,
	run_snapmogen_evaluation,
)
from utils.ae_t2m_eval import format_t2m_metrics, run_t2m_evaluation
from utils.define_device import define_device
from utils.define_num_workers import define_num_workers
from utils.load_t2m_encoders import load_t2m_motion_encoders
from utils.snapmogen_evaluator import (
	DEFAULT_EVALUATOR_DIR as SNAPMOGEN_EVALUATOR_DIR,
	compute_snapmogen_perceptual_features,
	load_snapmogen_evaluator,
)
from utils.training_related import save_to_log


if os.environ.get('DISPLAY', '') == '':
	matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_working_dir(base_dir, dataset, k, latent_dim, min_len, max_len):
	now = datetime.now()
	prefix = 'snap-alae' if dataset == 'snapmogen' else 'alae'
	run_name = (
		f"{now.strftime('%m%d%Y')}-{prefix}"
		f"-k{k}"
		f"-d{latent_dim}"
		f"-varlen{min_len}-{max_len}"
	)
	return os.path.join(base_dir, run_name)


def package_alae_checkpoint(state_dict, latent_stats=None):
	"""Self-contained ALAE checkpoint: weights + (optional) per-dim latent
	standardization stats for the variational LLM head, so downstream training
	reads everything from one file. Loaded by ALAEWrapper / ALAEMotionGPT.

	Format: ``{'state_dict': <weights>, 'latent_stats': {'mean','std','count'}}``.
	"""
	return {'state_dict': state_dict, 'latent_stats': latent_stats}


def append_t2m_metrics(csv_path, epoch, metrics):
	fieldnames = ['epoch', *metrics.keys()]
	file_exists = os.path.isfile(csv_path)
	with open(csv_path, 'a', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		row = {'epoch': epoch}
		row.update({key: float(value) for key, value in metrics.items()})
		writer.writerow(row)


def update_alae_plot(
	save_dir,
	hist_epoch,
	hist_train_rec,
	hist_val_rec,
	hist_train_ric,
	hist_val_ric,
	hist_train_percept,
	hist_val_percept,
	fig,
	ax,
	interactive_backend,
):
	ax.cla()
	ax.plot(hist_epoch, hist_train_rec, label='Train Rec (SmoothL1)', color='tab:blue', linewidth=1.8)
	ax.plot(hist_epoch, hist_val_rec, label='Val Rec', color='tab:orange', linewidth=1.8, linestyle='--')
	ax.plot(hist_epoch, hist_train_ric, label='Train RIC', color='tab:green', linewidth=1.4)
	ax.plot(hist_epoch, hist_val_ric, label='Val RIC', color='tab:olive', linewidth=1.4, linestyle='--')
	ax.plot(hist_epoch, hist_train_percept, label='Train Percept', color='tab:purple', linewidth=1.4)
	ax.plot(hist_epoch, hist_val_percept, label='Val Percept', color='tab:red', linewidth=1.4, linestyle='--')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss')
	ax.set_title('ALAE Reconstruction / RIC / Perceptual')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right', frameon=True, fontsize=8)
	fig.tight_layout()

	if interactive_backend:
		plt.pause(0.001)

	fig.savefig(os.path.join(save_dir, 'alae_training_curve.png'), dpi=150)


def build_optimizer(model, lr):
	with_decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if name.endswith('.bias') or 'norm' in name.lower() or 'bn' in name.lower():
			no_decay.append(param)
		else:
			with_decay.append(param)
	return optim.AdamW(
		[
			{'params': with_decay, 'lr': lr, 'weight_decay': 1e-4},
			{'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
		]
	)


def latent_decorr_loss(latents, target=0.0):
	"""Cross-slot latent decorrelation over batch-centered per-sample cosines.

	target <= 0 (2026-07-20 original): mean(cos^2) -- an unbounded push toward
	cos=0. The shipped K4 checkpoint encodes IDENTICAL latents for slots 0/1/3
	(centered cos 1.000 over the full test split), the root cause of every
	downstream slot-clone finding; squared cosine penalises both positive and
	negative correlation with a smooth gradient, 0 = decorrelated.

	target > 0 (2026-07-21 MID line): mean((cos^2 - target^2)^2) -- a two-sided
	SETPOINT at |cos| = target. Measured on the 07-20 decA run, decorrelation is
	nearly free (recon improved 33% while cos2 went to 0.0000 and stayed), so a
	one-sided form leaves everything below the target gradient-free and the pairs
	drift on to 0 anyway. The squared-deviation form makes |cos| = target an
	attractor from both sides, which is what "hold the slots ~50% similar" needs.
	Note the setpoint is on the MAGNITUDE: a pair may settle at cos = -target.
	"""
	lat = latents.float()
	K = lat.shape[1]
	if K < 2:
		return lat.sum() * 0.0
	c = lat - lat.mean(dim=0, keepdim=True)
	c = c / (c.norm(dim=-1, keepdim=True) + 1e-6)
	iu = torch.triu_indices(K, K, offset=1)
	cos = torch.einsum('bkd,bjd->bkj', c, c)[:, iu[0], iu[1]]
	sq = cos ** 2
	if target <= 0.0:
		return sq.mean()
	return ((sq - float(target) ** 2) ** 2).mean()


def main():
	parser = argparse.ArgumentParser(description='Adaptive-length AutoEncoder Training Script')
	parser.add_argument('--dataset', type=str, default='humanml3d', choices=['humanml3d', 'snapmogen'],
		help='Training dataset. humanml3d: 263-d feats @20fps. snapmogen: 296-d feats @30fps.')
	parser.add_argument('--data_root', type=str, default=None,
		help='Dataset root. Defaults per dataset: humanml3d -> datasets/humanml3d/, '
		     'snapmogen -> datasets/snapmogen/.')
	parser.add_argument('--working_dir', type=str, default='experiments/alae/')
	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--min_len', type=int, default=None,
		help='Drop motions shorter than this (frames). Defaults per dataset: humanml3d 20, '
		     'snapmogen 128 (official min_motion_length).')
	parser.add_argument('--max_len', type=int, default=None,
		help='Maximum training clip length (frames). Defaults per dataset: humanml3d 200 '
		     '(T2M eval upper bound), snapmogen 320 (official max_motion_length).')
	# Stage-1 capacity defaults.
	parser.add_argument('--k', type=int, default=4,
		help='Number of continuous latent slots the clip is compressed into. '
		     'The best budget is dataset-dependent; the released HumanML3D '
		     'model uses 2 and the SnapMoGen model uses 4.')
	parser.add_argument('--latent_dim', type=int, default=512)
	parser.add_argument('--hidden_dim', type=int, default=512)
	parser.add_argument('--depth', type=int, default=3)
	parser.add_argument('--num_res_blocks', type=int, default=2)
	parser.add_argument('--dilation_growth_rate', type=int, default=3)
	parser.add_argument('--num_encoder_layers', type=int, default=4)
	parser.add_argument('--num_decoder_layers', type=int, default=4)
	parser.add_argument('--nhead', type=int, default=8)
	parser.add_argument('--dim_feedforward', type=int, default=2048)
	parser.add_argument('--dropout', type=float, default=0.05)
	parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu', 'silu'])
	parser.add_argument('--norm', type=str, default=None, choices=[None, 'LN', 'GN', 'BN'])
	parser.add_argument('--num_workers', type=int, default=define_num_workers())
	parser.add_argument('--max_decode_len', type=int, default=256)
	# Phase 2 loss weights.
	parser.add_argument('--lambda_rec', type=float, default=1.0)
	parser.add_argument('--lambda_ric', type=float, default=0.5)
	parser.add_argument('--lambda_percept', type=float, default=10)
	parser.add_argument('--percept_cosine', action='store_true',
		help='Add a cosine-similarity term mean(1-cos) on the T2M perceptual '
		     'embeddings alongside the MSE term (direction + magnitude matching).')
	parser.add_argument('--lambda_percept_cosine', type=float, default=1.0,
		help='Weight for the perceptual cosine term (only used with --percept_cosine).')
	parser.add_argument('--lambda_latent_l2', type=float, default=0)
	parser.add_argument('--lambda_ortho', type=float, default=1,
		help='Weight for the latent_queries orthogonality loss.')
	parser.add_argument('--lambda_latent_decorr', type=float, default=0.0,
		help='Weight for the cross-slot latent decorrelation loss (mean over '
			'slot pairs of per-sample squared cosine between batch-centered '
			'latents). 0 = off, byte-identical baseline. 2026-07-20.')
	parser.add_argument('--decorr_target', type=float, default=0.0,
		help='Setpoint for the decorrelation loss, as |centered cos| between '
			'slot pairs. 0 (default) = the 2026-07-20 form, an unbounded push '
			'to cos=0. >0 switches to the two-sided form mean((cos^2-t^2)^2), '
			'which HOLDS pairwise similarity at t instead of removing it '
			'(2026-07-21 MID line; lambda alone is a speed knob, not a '
			'setpoint knob). Requires --lambda_latent_decorr > 0.')
	parser.add_argument('--disable_perceptual', action='store_true',
		help='Skip loading frozen perceptual encoders and zero out perceptual loss (ablation).')
	parser.add_argument('--t2m_ckpt', type=str,
		default=os.path.join('deps', 't2m', 't2m', 'text_mot_match', 'model', 'finest.tar'))
	parser.add_argument('--snapmogen_evaluator_dir', type=str, default=SNAPMOGEN_EVALUATOR_DIR,
		help='Directory of the official SnapMoGen evaluator (perceptual loss + metrics '
		     'on the snapmogen dataset). Run prepare/download_snapmogen_evaluator.sh.')
	parser.add_argument('--eval_t2m_every', type=int, default=5)
	parser.add_argument('--t2m_batch_size', type=int, default=None,
		help='Retrieval candidate pool per batch. Defaults per dataset: humanml3d 32 '
		     '(TM2T protocol), snapmogen 100 (official matching_pool_size).')
	parser.add_argument('--t2m_num_workers', type=int, default=0)
	parser.add_argument('--t2m_split', type=str, default='test', choices=['val', 'test'])
	parser.add_argument('--device', type=str, default=None,
		help="CUDA device to use, e.g. '0', 'cuda:1', 'cpu'. Defaults to auto-detect (cuda:0 if available).")
	args = parser.parse_args()

	# Resolve dataset-specific settings.
	is_snapmogen = args.dataset == 'snapmogen'
	if args.data_root is None:
		args.data_root = (
			'datasets/snapmogen/' if is_snapmogen
			else 'datasets/humanml3d/'
		)
	if args.min_len is None:
		args.min_len = 128 if is_snapmogen else 20
	if args.max_len is None:
		args.max_len = 320 if is_snapmogen else 200
	input_dim = 296 if is_snapmogen else 263
	# Explicit-position loss slice: SnapMoGen's official trainer supervises
	# [..., :148] on the 296-d feats (forward_attn); HumanML3D uses ric [4:67].
	ric_slice = slice(0, 148) if is_snapmogen else slice(4, 67)
	if args.t2m_batch_size is None:
		# Retrieval candidate pool: 32 for the HumanML3D TM2T protocol, 100 for
		# the SnapMoGen protocol (official matching_pool_size).
		args.t2m_batch_size = 100 if is_snapmogen else 32

	working_dir = build_working_dir(
		args.working_dir,
		dataset=args.dataset,
		k=args.k,
		latent_dim=args.latent_dim,
		min_len=args.min_len,
		max_len=args.max_len,
	)
	if os.path.exists(working_dir):
		shutil.rmtree(working_dir)
	os.makedirs(working_dir, exist_ok=True)

	device = define_device(args.device)
	amp_enabled = device.type == 'cuda'
	torch.backends.cudnn.benchmark = device.type == 'cuda'

	save_to_log(f'Using device: {device}', working_dir=working_dir, print_msg=True)
	save_to_log(f'Dataset: {args.dataset} (input_dim={input_dim}, root={args.data_root})',
		working_dir=working_dir, print_msg=True)
	if is_snapmogen:
		save_to_log(
			'SnapMoGen mode: perceptual loss + metrics use the official SnapMoGen evaluator '
			f'(TMR-style; cosine retrieval, pool={args.t2m_batch_size}; Matching_score is mean '
			f'cosine similarity, HIGHER is better); '
			f'explicit-position slice [{ric_slice.start}:{ric_slice.stop}].',
			working_dir=working_dir, print_msg=True,
		)
	save_to_log(f'Latent token length K: {args.k}', working_dir=working_dir, print_msg=True)
	save_to_log(f'Latent dim: {args.latent_dim}', working_dir=working_dir, print_msg=True)
	save_to_log(f'Train clip length range: [{args.min_len}, {args.max_len}]', working_dir=working_dir, print_msg=True)
	save_to_log(f'lambda_ortho (latent_queries orthogonality): {args.lambda_ortho}', working_dir=working_dir, print_msg=True)
	save_to_log(f'lambda_latent_decorr (cross-slot latent decorrelation): {args.lambda_latent_decorr}', working_dir=working_dir, print_msg=True)
	save_to_log(f'decorr_target (|centered cos| setpoint; 0 = push-to-zero): {args.decorr_target}', working_dir=working_dir, print_msg=True)
	if args.percept_cosine:
		save_to_log(
			f'Perceptual cosine term ENABLED (lambda_percept_cosine={args.lambda_percept_cosine}).',
			working_dir=working_dir, print_msg=True,
		)

	save_to_log('Loading dataset...', working_dir=working_dir, print_msg=True)
	dataset_cls = SnapMoGenVarLenDataset if is_snapmogen else HumanML3DVarLenDataset
	train_set = dataset_cls(
		data_root=args.data_root,
		split='train',
		min_len=args.min_len,
		max_len=args.max_len,
	)
	val_set = dataset_cls(
		data_root=args.data_root,
		split='val',
		min_len=args.min_len,
		max_len=args.max_len,
	)

	train_loader = DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=True,
		persistent_workers=args.num_workers > 0,
		collate_fn=varlen_collate,
	)
	val_loader = DataLoader(
		val_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=True,
		persistent_workers=args.num_workers > 0,
		collate_fn=varlen_collate,
	)

	model = AdaptiveLengthAutoEncoder(
		input_dim=input_dim,
		k=args.k,
		latent_dim=args.latent_dim,
		hidden_dim=args.hidden_dim,
		depth=args.depth,
		dilation_growth_rate=args.dilation_growth_rate,
		activation=args.activation,
		norm=args.norm,
		num_res_blocks=args.num_res_blocks,
		num_encoder_layers=args.num_encoder_layers,
		num_decoder_layers=args.num_decoder_layers,
		nhead=args.nhead,
		dim_feedforward=args.dim_feedforward,
		dropout=args.dropout,
		max_decode_len=max(args.max_decode_len, args.max_len),
	).to(device)

	# Phase 2: load frozen perceptual encoders.
	# - humanml3d: TM2T MovementConvEncoder + MotionEncoderBiGRUCo pair.
	# - snapmogen: official SnapMoGen evaluator (motion tower fid_emb doubles as
	#   the perceptual feature; the text tower/T5 is lazy-loaded only at eval).
	t2m_move_enc = None
	t2m_motion_enc = None
	snap_evaluator = None
	percept_fn = None
	if is_snapmogen:
		if not args.disable_perceptual or args.eval_t2m_every > 0:
			save_to_log(f'Loading frozen SnapMoGen evaluator from {args.snapmogen_evaluator_dir} ...',
				working_dir=working_dir, print_msg=True)
			snap_evaluator = load_snapmogen_evaluator(
				device=device, ckpt_dir=args.snapmogen_evaluator_dir,
			)
		if args.disable_perceptual:
			save_to_log('Perceptual loss DISABLED (--disable_perceptual).', working_dir=working_dir, print_msg=True)
		else:
			def percept_fn(motion, lengths):
				return compute_snapmogen_perceptual_features(motion, lengths, snap_evaluator)
			n_p = sum(p.numel() for p in snap_evaluator.latent_enc.parameters())
			save_to_log(
				f'SnapMoGen perceptual encoder ready (frozen motion tower, {n_p/1e6:.2f}M params); '
				f'lambda_ric={args.lambda_ric}, lambda_percept={args.lambda_percept}.',
				working_dir=working_dir, print_msg=True,
			)
	elif args.disable_perceptual:
		save_to_log('Perceptual loss DISABLED (--disable_perceptual).', working_dir=working_dir, print_msg=True)
	else:
		save_to_log(f'Loading frozen T2M perceptual encoders from {args.t2m_ckpt} ...',
			working_dir=working_dir, print_msg=True)
		t2m_move_enc, t2m_motion_enc = load_t2m_motion_encoders(
			device=device, checkpoint_path=args.t2m_ckpt,
		)
		n_p = sum(p.numel() for p in t2m_move_enc.parameters()) + sum(p.numel() for p in t2m_motion_enc.parameters())
		save_to_log(
			f'T2M perceptual encoders loaded (frozen, {n_p/1e6:.2f}M params); '
			f'lambda_ric={args.lambda_ric}, lambda_percept={args.lambda_percept}.',
			working_dir=working_dir, print_msg=True,
		)

	n_ae = sum(p.numel() for p in model.parameters() if p.requires_grad)
	save_to_log(f'AE trainable params: {n_ae/1e6:.2f}M', working_dir=working_dir, print_msg=True)

	optimizer = build_optimizer(model, args.lr)
	total_steps = len(train_loader) * args.num_epochs
	warmup_steps = int(0.03 * total_steps)
	scheduler = get_cosine_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)
	scaler = GradScaler(enabled=amp_enabled)

	plt.ion()
	interactive_backend = matplotlib.get_backend().lower() != 'agg'
	fig, ax = plt.subplots(figsize=(8, 5))

	hist_epoch = []
	hist_train_rec = []
	hist_val_rec = []
	hist_train_ric = []
	hist_val_ric = []
	hist_train_percept = []
	hist_val_percept = []

	# Track two separate "best" checkpoints:
	# - best_loss.pt: lowest validation ValLoss
	# - best_tm2t.pt: best TM2T metric (lowest FID) among T2M eval epochs
	best_loss = float('inf')
	best_loss_epoch = -1
	best_tm2t_fid = float('inf')
	best_tm2t_epoch = -1
	best_tm2t_metrics = None
	if is_snapmogen:
		# SnapMoGen checkpoint naming convention: snap-alae-k{K}.pt (best val
		# loss) and snap-alae-k{K}_tm2t.pt (best FID under the SnapMoGen
		# evaluator protocol).
		best_loss_path = os.path.join(working_dir, f'snap-alae-k{args.k}.pt')
		best_tm2t_path = os.path.join(working_dir, f'snap-alae-k{args.k}_tm2t.pt')
	else:
		best_loss_path = os.path.join(working_dir, 'best_loss.pt')
		best_tm2t_path = os.path.join(working_dir, 'best_tm2t.pt')
	t2m_csv_path = os.path.join(working_dir, 't2m_metrics.csv')
	# Built lazily on the first periodic eval so the feature preload only
	# happens when metrics actually run (snapmogen only).
	snap_eval_dataset = None

	save_to_log('Starting training loop...', working_dir=working_dir, print_msg=True)

	for epoch in range(args.num_epochs):
		model.train()
		train_loss_total = 0.0
		train_loss_rec = 0.0
		train_loss_ric = 0.0
		train_loss_percept = 0.0
		train_loss_latent_reg = 0.0
		train_loss_ortho = 0.0
		train_loss_decorr = 0.0
		train_latent_norm = 0.0

		progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}')
		for step, (batch, mask, lengths) in enumerate(progress_bar):
			batch = batch.to(device, non_blocking=True)
			mask = mask.to(device, non_blocking=True)
			lengths = lengths.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)

			with autocast(device_type=device.type, enabled=amp_enabled):
				x_hat, latents, memory = model(batch, target_len=batch.shape[1])
				losses = model.compute_loss(
					batch, x_hat, latents,
					mask=mask,
					lengths=lengths,
					t2m_move_enc=t2m_move_enc,
					t2m_motion_enc=t2m_motion_enc,
					percept_fn=percept_fn,
					lambda_rec=args.lambda_rec,
					lambda_ric=args.lambda_ric,
					lambda_percept=args.lambda_percept,
					lambda_latent_l2=args.lambda_latent_l2,
					lambda_ortho=args.lambda_ortho,
					ric_slice=ric_slice,
					use_percept_cosine=args.percept_cosine,
					lambda_percept_cosine=args.lambda_percept_cosine,
				)
				loss = losses['loss']

			loss_decorr = None
			if args.lambda_latent_decorr != 0.0:
				loss_decorr = latent_decorr_loss(latents, target=args.decorr_target)
				loss = loss + args.lambda_latent_decorr * loss_decorr

			if torch.isnan(loss) or torch.isinf(loss):
				save_to_log(
					f'Warning: NaN/Inf loss detected at epoch {epoch + 1}, batch {step}. Skipping batch.',
					working_dir=working_dir,
					print_msg=True,
				)
				optimizer.zero_grad(set_to_none=True)
				continue

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			train_loss_total += loss.item()
			train_loss_rec += losses['loss_rec'].item()
			train_loss_ric += losses['loss_ric'].item()
			train_loss_percept += losses['loss_percept'].item()
			train_loss_latent_reg += losses['loss_latent_l2'].item()
			train_loss_ortho += losses['loss_ortho'].item()
			train_loss_decorr += loss_decorr.item() if loss_decorr is not None else 0.0
			train_latent_norm += latents.norm(dim=-1).mean().item()

			postfix = {
				'lr': f"{scheduler.get_last_lr()[0]:.2e}",
				'rec': f"{losses['loss_rec'].item():.4f}",
				'ric': f"{losses['loss_ric'].item():.4f}",
				'pcp': f"{losses['loss_percept'].item():.4f}",
				'orth': f"{losses['loss_ortho'].item():.4f}",
				'lat_n': f"{latents.norm(dim=-1).mean().item():.2f}",
			}
			if loss_decorr is not None:
				postfix['dec'] = f'{loss_decorr.item():.4f}'
			if args.percept_cosine:
				postfix['pcp_cos'] = f"{losses['loss_percept_cos'].item():.4f}"
			progress_bar.set_postfix(postfix)

		avg_train_loss = train_loss_total / len(train_loader)
		avg_train_rec = train_loss_rec / len(train_loader)
		avg_train_ric = train_loss_ric / len(train_loader)
		avg_train_percept = train_loss_percept / len(train_loader)
		avg_train_latent_reg = train_loss_latent_reg / len(train_loader)
		avg_train_ortho = train_loss_ortho / len(train_loader)
		avg_train_latent_norm = train_latent_norm / len(train_loader)

		model.eval()
		val_loss_total = 0.0
		val_loss_rec = 0.0
		val_loss_ric = 0.0
		val_loss_percept = 0.0
		val_loss_latent_reg = 0.0
		val_loss_ortho = 0.0
		val_latent_norm = 0.0
		# Per-dim latent sum / sumsq, accumulated over the eval-mode val pass, to
		# embed standardization stats into the saved checkpoint. Eval mode (no
		# dropout) matches how the variational LLM head consumes encode_latents().
		val_lat_sum = None
		val_lat_sumsq = None
		val_lat_count = 0
		val_pair_cos_sum = None
		val_pair_sq_sum = 0.0
		val_pair_batches = 0
		with torch.no_grad():
			for batch, mask, lengths in val_loader:
				batch = batch.to(device, non_blocking=True)
				mask = mask.to(device, non_blocking=True)
				lengths = lengths.to(device, non_blocking=True)
				with autocast(device_type=device.type, enabled=amp_enabled):
					x_hat, latents, memory = model(batch, target_len=batch.shape[1])
					losses = model.compute_loss(
						batch, x_hat, latents,
						mask=mask,
						lengths=lengths,
						t2m_move_enc=t2m_move_enc,
						t2m_motion_enc=t2m_motion_enc,
						percept_fn=percept_fn,
						lambda_rec=args.lambda_rec,
						lambda_ric=args.lambda_ric,
						lambda_percept=args.lambda_percept,
						lambda_latent_l2=args.lambda_latent_l2,
						lambda_ortho=args.lambda_ortho,
						ric_slice=ric_slice,
						use_percept_cosine=args.percept_cosine,
						lambda_percept_cosine=args.lambda_percept_cosine,
					)

				val_loss_total += losses['loss'].item()
				val_loss_rec += losses['loss_rec'].item()
				val_loss_ric += losses['loss_ric'].item()
				val_loss_percept += losses['loss_percept'].item()
				val_loss_latent_reg += losses['loss_latent_l2'].item()
				val_loss_ortho += losses['loss_ortho'].item()
				val_latent_norm += latents.norm(dim=-1).mean().item()

				lat = latents.detach().float()              # (B, K, D)
				if val_lat_sum is None:
					val_lat_sum = lat.sum(dim=0)            # (K, D)
					val_lat_sumsq = (lat * lat).sum(dim=0)  # (K, D)
				else:
					val_lat_sum += lat.sum(dim=0)
					val_lat_sumsq += (lat * lat).sum(dim=0)
				val_lat_count += int(lat.shape[0])
				cvec = lat - lat.mean(dim=0, keepdim=True)
				cvec = cvec / (cvec.norm(dim=-1, keepdim=True) + 1e-6)
				iu_v = torch.triu_indices(lat.shape[1], lat.shape[1], offset=1)
				pc = torch.einsum('bkd,bjd->bkj', cvec, cvec)[:, iu_v[0], iu_v[1]].mean(dim=0)
				val_pair_sq_sum += float((pc ** 2).mean())
				val_pair_cos_sum = pc.detach().cpu() if val_pair_cos_sum is None else val_pair_cos_sum + pc.detach().cpu()
				val_pair_batches += 1

		avg_val_loss = val_loss_total / len(val_loader)
		avg_val_rec = val_loss_rec / len(val_loader)
		avg_val_ric = val_loss_ric / len(val_loader)
		avg_val_percept = val_loss_percept / len(val_loader)
		avg_val_latent_reg = val_loss_latent_reg / len(val_loader)
		avg_val_ortho = val_loss_ortho / len(val_loader)
		avg_val_latent_norm = val_latent_norm / len(val_loader)
		if val_pair_cos_sum is not None and val_pair_batches > 0:
			_pm = [round(float(v), 3) for v in (val_pair_cos_sum / val_pair_batches)]
			save_to_log(f'[decorr] epoch {epoch + 1}: val mean cos2={val_pair_sq_sum / val_pair_batches:.4f}; per-pair centered cos={_pm} (upper-tri order)', working_dir=working_dir, print_msg=True)

		# Finalize this epoch's latent standardization stats (embedded into any
		# checkpoint saved below).
		latent_stats = None
		if val_lat_count >= 2:
			mean_kd = val_lat_sum / val_lat_count
			var_kd = (val_lat_sumsq / val_lat_count) - mean_kd * mean_kd
			latent_stats = {
				'mean': mean_kd.detach().cpu(),
				'std': var_kd.clamp_min(0.0).sqrt().detach().cpu(),
				'count': int(val_lat_count),
			}

		if avg_val_loss < best_loss:
			best_loss = avg_val_loss
			best_loss_epoch = epoch + 1
			best_loss_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			torch.save(package_alae_checkpoint(best_loss_state, latent_stats), best_loss_path)
			save_to_log(
				f'[Best ValLoss] Epoch {best_loss_epoch} | ValLoss={best_loss:.6f} -> {best_loss_path}',
				working_dir=working_dir,
				print_msg=True,
			)

		save_to_log(f'---- Epoch {epoch + 1} Summary ----', working_dir=working_dir, print_msg=True)
		save_to_log(
			(
				f'Training:   Total: {avg_train_loss:.4f}, Rec: {avg_train_rec:.4f}, '
				f'RIC: {avg_train_ric:.4f}, Percept: {avg_train_percept:.4f}, '
				f'Ortho: {avg_train_ortho:.4f}, '
				f'LatReg: {avg_train_latent_reg:.4f}, LatNorm: {avg_train_latent_norm:.3f}'
			),
			working_dir=working_dir,
			print_msg=True,
		)
		save_to_log(
			(
				f'Validation: Total: {avg_val_loss:.4f}, Rec: {avg_val_rec:.4f}, '
				f'RIC: {avg_val_ric:.4f}, Percept: {avg_val_percept:.4f}, '
				f'Ortho: {avg_val_ortho:.4f}, '
				f'LatReg: {avg_val_latent_reg:.4f}, LatNorm: {avg_val_latent_norm:.3f}'
			),
			working_dir=working_dir,
			print_msg=True,
		)

		if args.eval_t2m_every > 0 and (epoch + 1) % args.eval_t2m_every == 0:
			try:
				if is_snapmogen:
					if snap_eval_dataset is None:
						snap_eval_dataset = SnapMoGenTextMotionEvalDataset(
							data_root=args.data_root,
							split=args.t2m_split,
							min_len=args.min_len,
							max_len=args.max_len,
						)
					t2m_metrics = run_snapmogen_evaluation(
						ae_model=model,
						evaluator=snap_evaluator,
						dataset=snap_eval_dataset,
						device=device,
						batch_size=args.t2m_batch_size,
						num_workers=args.t2m_num_workers,
					)
				else:
					t2m_metrics = run_t2m_evaluation(
						ae_model=model,
						data_root=args.data_root,
						batch_size=args.t2m_batch_size,
						num_workers=args.t2m_num_workers,
						split=args.t2m_split,
						device=device,
						output_dir=working_dir,
					)
				append_t2m_metrics(t2m_csv_path, epoch + 1, t2m_metrics)
				save_to_log(
					f'[T2M@epoch {epoch + 1}] {format_t2m_metrics(t2m_metrics)}',
					working_dir=working_dir,
					print_msg=True,
				)
				current_fid = float(t2m_metrics.get('FID', float('inf')))
				if current_fid < best_tm2t_fid:
					best_tm2t_fid = current_fid
					best_tm2t_epoch = epoch + 1
					best_tm2t_metrics = {k: float(v) for k, v in t2m_metrics.items()}
					best_tm2t_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
					torch.save(package_alae_checkpoint(best_tm2t_state, latent_stats), best_tm2t_path)
					save_to_log(
						f'[Best TM2T] Epoch {best_tm2t_epoch} | FID={best_tm2t_fid:.4f} -> {best_tm2t_path}',
						working_dir=working_dir,
						print_msg=True,
					)
			except Exception as exc:
				save_to_log(
					f'[T2M@epoch {epoch + 1}] FAILED: {type(exc).__name__}: {exc}',
					working_dir=working_dir,
					print_msg=True,
				)
		save_to_log('----------------------------\n', working_dir=working_dir, print_msg=True)

		hist_epoch.append(epoch + 1)
		hist_train_rec.append(avg_train_rec)
		hist_val_rec.append(avg_val_rec)
		hist_train_ric.append(avg_train_ric)
		hist_val_ric.append(avg_val_ric)
		hist_train_percept.append(avg_train_percept)
		hist_val_percept.append(avg_val_percept)
		update_alae_plot(
			save_dir=working_dir,
			hist_epoch=hist_epoch,
			hist_train_rec=hist_train_rec,
			hist_val_rec=hist_val_rec,
			hist_train_ric=hist_train_ric,
			hist_val_ric=hist_val_ric,
			hist_train_percept=hist_train_percept,
			hist_val_percept=hist_val_percept,
			fig=fig,
			ax=ax,
			interactive_backend=interactive_backend,
		)

	save_to_log('Training complete!', working_dir=working_dir, print_msg=True)
	if best_loss_epoch > 0:
		save_to_log(
			f'Best ValLoss ckpt: epoch {best_loss_epoch}, val_loss={best_loss:.6f} -> {best_loss_path}',
			working_dir=working_dir,
			print_msg=True,
		)
	else:
		save_to_log('No best_loss checkpoint was captured.', working_dir=working_dir, print_msg=True)
	if best_tm2t_epoch > 0 and best_tm2t_metrics is not None:
		save_to_log(
			f'Best TM2T ckpt: epoch {best_tm2t_epoch}, FID={best_tm2t_fid:.4f} -> {best_tm2t_path}',
			working_dir=working_dir,
			print_msg=True,
		)
	else:
		save_to_log('No best_tm2t checkpoint was captured (T2M eval never ran or always failed).',
			working_dir=working_dir, print_msg=True)


if __name__ == '__main__':
	main()
