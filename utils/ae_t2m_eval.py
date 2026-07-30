from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from motGPT.config import get_module_config
from motGPT.data.build_data import build_data
from motGPT.metrics.base import BaseMetrics


class NoopMotionVAE(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()

	def forward(self, motion):
		return motion


def load_t2m_eval_cfg(data_root, batch_size, num_workers, split, output_dir):
	try:
		OmegaConf.register_new_resolver("eval", eval)
	except ValueError:
		pass

	cfg_assets = OmegaConf.load("configs/assets.yaml")
	cfg_base = OmegaConf.load("configs/default.yaml")
	cfg_base = get_module_config(cfg_base, "configs")
	cfg = OmegaConf.merge(cfg_base, cfg_assets)

	cfg.DEBUG = False
	cfg.FOLDER = output_dir
	cfg.FOLDER_EXP = output_dir
	cfg.TIME = "ae_t2m_eval"
	cfg.TRAIN.STAGE = "lm_pretrain"
	cfg.TRAIN.instruction_type = "t2m"
	cfg.TEST.SPLIT = split
	cfg.TEST.BATCH_SIZE = batch_size
	cfg.TEST.NUM_WORKERS = num_workers
	cfg.TEST.REPLICATION_TIMES = 1
	cfg.TEST.SAVE_PREDICTIONS = False
	cfg.DATASET.target = "motGPT.data.HumanML3D.HumanML3DDataModule"
	cfg.DATASET.CODE_PATH = ""
	cfg.DATASET.TASK_PATH = ""
	cfg.DATASET.HUMANML3D.ROOT = data_root
	cfg.DATASET.HUMANML3D.SPLIT_ROOT = data_root
	cfg.DATASET.NFEATS = 263
	cfg.model.target = "motGPT.models.mgpt.MotionGPT"
	cfg.model.params.task = "t2m"
	cfg.model.params.metrics_dict = ["TM2TMetrics"]
	cfg.model.params.motion_vae = OmegaConf.create(
		{
			"target": "utils.ae_t2m_eval.NoopMotionVAE",
			"params": {
				"code_num": 1,
			},
		}
	)
	cfg.METRIC.TYPE = ["TM2TMetrics"]
	return cfg


def _get_batch_value(batch, key, default=None):
	if isinstance(batch, dict):
		return batch.get(key, default)
	return default


def _extract_motion(batch):
	motion = _get_batch_value(batch, "motion")
	if motion is not None:
		return motion
	if isinstance(batch, (tuple, list)) and len(batch) > 3:
		return batch[3]
	raise KeyError("motion")


def _extract_lengths(batch, motion=None):
	if isinstance(batch, dict):
		for key in ("length", "lengths", "m_length", "motion_length", "motion_len"):
			value = batch.get(key)
			if value is not None:
				if torch.is_tensor(value):
					return [int(length) for length in value.tolist()]
				return [int(length) for length in value]
	elif isinstance(batch, (tuple, list)) and len(batch) > 4:
		value = batch[4]
		if torch.is_tensor(value):
			return [int(length) for length in value.tolist()]
		return [int(length) for length in value]

	if motion is None:
		motion = _extract_motion(batch)
	if torch.is_tensor(motion):
		return [int(motion.shape[1])] * int(motion.shape[0])
	raise KeyError("length")


def _extract_reconstruction(model_output):
	if torch.is_tensor(model_output):
		return model_output
	if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
		first = model_output[0]
		if torch.is_tensor(first):
			return first
	raise TypeError(f"Unsupported autoencoder output type: {type(model_output)!r}")


class AETwitterAdapter(pl.LightningModule):
	def __init__(self, ae_model, datamodule, cfg):
		super().__init__()
		self.ae_model = ae_model
		self.datamodule = datamodule
		self.cfg = cfg
		self.metrics = BaseMetrics(cfg=cfg, datamodule=datamodule, debug=cfg.DEBUG)
		self.hparams_stub = SimpleNamespace(metrics_dict=["TM2TMetrics"])
		self.is_mm = False

	def forward(self, batch):
		feats_ref = _extract_motion(batch)
		feats_rst = _extract_reconstruction(self.ae_model(feats_ref))
		return feats_rst

	@torch.no_grad()
	def val_t2m_forward(self, batch):
		feats_ref = _extract_motion(batch)
		lengths = _extract_lengths(batch, motion=feats_ref)
		feats_rst = _extract_reconstruction(self.ae_model(feats_ref))

		joints_ref = self.datamodule.feats2joints(feats_ref)
		joints_rst = self.datamodule.feats2joints(feats_rst)

		feats_ref_r = self.datamodule.renorm4t2m(feats_ref)
		feats_rst_r = self.datamodule.renorm4t2m(feats_rst)

		return {
			"m_ref": feats_ref_r,
			"m_rst": feats_rst_r,
			"joints_ref": joints_ref,
			"joints_rst": joints_rst,
			"length": lengths,
		}

	@torch.no_grad()
	def validation_step(self, batch, batch_idx):
		rs_set = self.val_t2m_forward(batch)
		word_embs = _get_batch_value(batch, "word_embs")
		pos_ohot = _get_batch_value(batch, "pos_ohot")
		text_lengths = _get_batch_value(batch, "text_len")
		lengths = _extract_lengths(batch)

		self.metrics.TM2TMetrics.update(
			feats_ref=rs_set["m_ref"],
			feats_rst=rs_set["m_rst"],
			lengths_ref=lengths,
			lengths_rst=rs_set["length"],
			word_embs=word_embs,
			pos_ohot=pos_ohot,
			text_lengths=text_lengths,
		)

	def on_validation_epoch_end(self):
		metric_dict = self.metrics.TM2TMetrics.compute(sanity_flag=self.trainer.sanity_checking)
		log_dict = {}
		for key, value in metric_dict.items():
			if torch.is_tensor(value):
				value = value.item()
			log_dict[f"Metrics/TM2TMetrics_{key}"] = float(value)
		if log_dict and not self.trainer.sanity_checking:
			self.log_dict(log_dict, sync_dist=False, rank_zero_only=True)


def run_t2m_evaluation(ae_model, data_root, batch_size, num_workers, split, device, output_dir):
	cfg = load_t2m_eval_cfg(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
		split=split,
		output_dir=output_dir,
	)
	datamodule = build_data(cfg)
	eval_model = AETwitterAdapter(ae_model=ae_model, datamodule=datamodule, cfg=cfg)
	eval_model.to(device).eval()

	trainer = pl.Trainer(
		accelerator="gpu" if device.type == "cuda" else "cpu",
		devices=1,
		logger=False,
		enable_progress_bar=True,
		enable_model_summary=False,
		enable_checkpointing=False,
		deterministic=False,
	)
	was_training = ae_model.training
	try:
		results = trainer.validate(eval_model, datamodule=datamodule, verbose=False)
	finally:
		ae_model.to(device)
		if was_training:
			ae_model.train()
		else:
			ae_model.eval()
	if not results:
		return {}

	metrics = {}
	for key, value in results[0].items():
		short_key = key.split("TM2TMetrics_", 1)[-1] if "TM2TMetrics_" in key else key
		metrics[short_key] = float(value)
	return metrics


def format_t2m_metrics(metrics):
	keys_priority = [
		"R_precision_top_1",
		"R_precision_top_2",
		"R_precision_top_3",
		"gt_R_precision_top_1",
		"gt_R_precision_top_2",
		"gt_R_precision_top_3",
		"FID",
		"Matching_score",
		"gt_Matching_score",
		"Diversity",
		"gt_Diversity",
	]
	parts = []
	for key in keys_priority:
		if key in metrics:
			parts.append(f"{key}={metrics[key]:.4f}")
	for key, value in metrics.items():
		if key not in keys_priority:
			parts.append(f"{key}={value:.4f}")
	return " | ".join(parts) if parts else "<no metrics>"