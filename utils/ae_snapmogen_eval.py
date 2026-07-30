"""SnapMoGen reconstruction evaluation following the official protocol.

Mirrors ``evaluation_vqvae`` from snap-research/SnapMoGen (``utils/eval_t2m.py``):
GT and AE-reconstructed motions are embedded by the frozen SnapMoGen evaluator;
FID / Diversity come from ``fid_emb``, R-precision / Matching from cosine
similarity between text and motion ``mu`` embeddings over a candidate pool of
one batch (official ``matching_pool_size`` = 100). The metric helpers are
verbatim ports of the official ``utils/metrics.py``.

NB: ``Matching_score`` here is mean cosine similarity -- HIGHER is better
(opposite direction of the HumanML3D MMDist/Matching score).
"""

import json
import random
from os.path import join as pjoin

import numpy as np
import torch
from scipy import linalg
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric helpers (ported from snap-research/SnapMoGen utils/metrics.py).
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(matrix1, matrix2):
	matrix1 = matrix1 / np.linalg.norm(matrix1, axis=-1, keepdims=True)
	matrix2 = matrix2 / np.linalg.norm(matrix2, axis=-1, keepdims=True)
	return np.dot(matrix1, matrix2.T)


def calculate_top_k(mat, top_k):
	size = mat.shape[0]
	gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
	bool_mat = (mat == gt_mat)
	correct_vec = False
	top_k_list = []
	for i in range(top_k):
		correct_vec = (correct_vec | bool_mat[:, i])
		top_k_list.append(correct_vec[:, None])
	return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
	"""Cosine-similarity R-precision (the SnapMoGen protocol variant)."""
	dist_mat = -cosine_similarity_matrix(embedding1, embedding2)
	argmax = np.argsort(dist_mat, axis=1)
	top_k_mat = calculate_top_k(argmax, top_k)
	return top_k_mat.sum(axis=0) if sum_all else top_k_mat


def calculate_activation_statistics(activations):
	mu = np.mean(activations, axis=0)
	cov = np.cov(activations, rowvar=False)
	return mu, cov


def calculate_diversity(activation, diversity_times):
	assert len(activation.shape) == 2
	assert activation.shape[0] > diversity_times
	num_samples = activation.shape[0]
	first_indices = np.random.choice(num_samples, diversity_times, replace=False)
	second_indices = np.random.choice(num_samples, diversity_times, replace=False)
	dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
	return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Frechet Distance between N(mu1, sigma1) and N(mu2, sigma2)."""
	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)
	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)
	assert mu1.shape == mu2.shape
	assert sigma1.shape == sigma2.shape

	diff = mu1 - mu2
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		print(f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates')
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
		covmean = covmean.real
	return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ---------------------------------------------------------------------------
# Eval dataset: official TextMotionDataset recipe (caption + padded motion).
# ---------------------------------------------------------------------------

class SnapMoGenTextMotionEvalDataset(data.Dataset):
	"""Text-motion pairs for evaluator-based metrics.

	Mirrors the official ``TextMotionDataset``: clips below ``min_len`` are
	dropped, the caption is sampled from manual+gpt, the motion is cropped to a
	``unit_length`` multiple (capped at ``max_len``) at a random offset and
	zero-padded to ``max_len``. Motions are z-normalized with the official
	mean/std, matching the evaluator's training input space.
	"""

	def __init__(
		self,
		data_root,
		split='test',
		min_len=128,
		max_len=320,
		unit_length=8,
		features_dim=296,
	):
		self.min_len = int(min_len)
		self.max_len = int(max_len)
		self.unit_length = int(unit_length)

		feat_dir = pjoin(data_root, 'renamed_feats')
		meta_dir = pjoin(data_root, 'meta_data')
		split_dir = pjoin(data_root, 'data_split_info')

		self.mean = np.load(pjoin(meta_dir, 'mean.npy')).astype(np.float32)
		self.std = np.load(pjoin(meta_dir, 'std.npy')).astype(np.float32)
		if self.mean.shape[0] != features_dim:
			raise ValueError(
				f'Dimension mismatch - mean has {self.mean.shape[0]}, expected {features_dim}')

		with open(pjoin(data_root, 'all_caption_clean.json'), 'r') as f:
			self.all_captions = json.load(f)

		clips = []
		with open(pjoin(split_dir, f'{split}_ids.txt'), 'r') as f:
			for line in f:
				cid = line.strip()
				if not cid:
					continue
				mid, start, end = cid.split('#')
				if int(end) - int(start) < self.min_len:
					continue
				clips.append((cid, mid, int(start), int(end)))

		needed_mids = {mid for _, mid, _, _ in clips}
		print(f'[SnapMoGen Eval] Loading {split}: {len(needed_mids)} feature files ...')
		self.data_dict = {}
		for mid in tqdm(sorted(needed_mids)):
			try:
				self.data_dict[mid] = np.load(pjoin(feat_dir, f'{mid}.npy')).astype(np.float32)
			except Exception:
				pass
		self.clips = [c for c in clips if c[1] in self.data_dict]
		print(f'[SnapMoGen Eval] Loaded {len(self.clips)} text-motion clips.')

	def __len__(self):
		return len(self.clips)

	def __getitem__(self, item):
		cid, mid, start, end = self.clips[item]
		motion = self.data_dict[mid][start:end]
		motion = (motion - self.mean) / self.std

		captions = self.all_captions[cid]['manual'] + self.all_captions[cid]['gpt']
		caption = random.choice(captions)

		m_length = min(len(motion), self.max_len)
		m_length = (m_length // self.unit_length) * self.unit_length
		idx = random.randint(0, len(motion) - m_length)
		motion = motion[idx: idx + m_length]
		if m_length < self.max_len:
			motion = np.concatenate(
				[motion, np.zeros((self.max_len - m_length, motion.shape[1]), dtype=np.float32)],
				axis=0,
			)
		return caption, torch.from_numpy(np.ascontiguousarray(motion)).float(), m_length


# ---------------------------------------------------------------------------
# Reconstruction evaluation (official evaluation_vqvae, minus MPJPE).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_snapmogen_evaluation(
	ae_model,
	evaluator,
	data_root=None,
	dataset=None,
	device=None,
	batch_size=100,
	num_workers=0,
	split='test',
	max_batches=None,
):
	"""Evaluate AE reconstructions with the SnapMoGen evaluator.

	``batch_size`` is the R-precision candidate pool (official
	``matching_pool_size`` = 100; not comparable to the 32-pool HumanML3D
	protocol). Pass a prebuilt ``dataset`` to avoid reloading features on
	every periodic eval. Returns a dict whose keys line up with
	``format_t2m_metrics`` / ``append_t2m_metrics``.
	"""
	if dataset is None:
		if data_root is None:
			raise ValueError('Provide either data_root or a prebuilt dataset.')
		dataset = SnapMoGenTextMotionEvalDataset(data_root=data_root, split=split)
	if device is None:
		device = next(ae_model.parameters()).device

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True,
	)

	was_training = ae_model.training
	ae_model.eval()

	motion_annotation_list = []
	motion_pred_list = []
	R_precision_real = 0
	R_precision = 0
	matching_score_real = 0.0
	matching_score_pred = 0.0
	nb_sample = 0

	try:
		for batch_idx, (texts, motions, m_lengths) in enumerate(tqdm(loader, desc='SnapMoGen eval')):
			if max_batches is not None and batch_idx >= max_batches:
				break
			motions = motions.to(device, non_blocking=True).float()
			m_lengths = m_lengths.to(device).long()

			et = evaluator.encode_text(texts).cpu().numpy()
			fid_em, em = evaluator.encode_motion(motions, m_lengths)

			x_hat, _, _ = ae_model(motions, target_len=motions.shape[1])
			fid_em_pred, em_pred = evaluator.encode_motion(x_hat.float(), m_lengths)

			motion_annotation_list.append(fid_em.cpu())
			motion_pred_list.append(fid_em_pred.cpu())

			em = em.cpu().numpy()
			em_pred = em_pred.cpu().numpy()
			R_precision_real += calculate_R_precision(et, em, top_k=3, sum_all=True)
			matching_score_real += cosine_similarity_matrix(et, em).trace()
			R_precision += calculate_R_precision(et, em_pred, top_k=3, sum_all=True)
			matching_score_pred += cosine_similarity_matrix(et, em_pred).trace()
			nb_sample += motions.shape[0]
	finally:
		if was_training:
			ae_model.train()

	if nb_sample == 0:
		return {}

	motion_annotation_np = torch.cat(motion_annotation_list, dim=0).numpy()
	motion_pred_np = torch.cat(motion_pred_list, dim=0).numpy()
	gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
	mu, cov = calculate_activation_statistics(motion_pred_np)
	fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

	diversity_times = 300 if nb_sample > 300 else min(100, nb_sample - 1)
	diversity_real = calculate_diversity(motion_annotation_np, diversity_times)
	diversity = calculate_diversity(motion_pred_np, diversity_times)

	R_precision_real = R_precision_real / nb_sample
	R_precision = R_precision / nb_sample

	return {
		'FID': float(fid),
		'R_precision_top_1': float(R_precision[0]),
		'R_precision_top_2': float(R_precision[1]),
		'R_precision_top_3': float(R_precision[2]),
		'gt_R_precision_top_1': float(R_precision_real[0]),
		'gt_R_precision_top_2': float(R_precision_real[1]),
		'gt_R_precision_top_3': float(R_precision_real[2]),
		'Matching_score': float(matching_score_pred / nb_sample),
		'gt_Matching_score': float(matching_score_real / nb_sample),
		'Diversity': float(diversity),
		'gt_Diversity': float(diversity_real),
	}
