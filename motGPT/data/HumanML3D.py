import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import (
    Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, 
    Text2MotionDatasetCBV3, Text2MotionDatasetEvalV3
)
from .utils import humanml3d_collate


class HumanML3DDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22
        self.fps = cfg.DATASET.HUMANML3D.FPS
        self.hparams.fps = cfg.DATASET.HUMANML3D.FPS
        
        # Path to the dataset
        data_root = cfg.DATASET.HUMANML3D.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')
        self.hparams.instruction_type = cfg.TRAIN.instruction_type

        motion_vae_cfg = cfg.model.params.get('motion_vae', None)
        motion_vae_target = None
        if motion_vae_cfg is not None:
            motion_vae_target = motion_vae_cfg.get('target', None)
        uses_vqvae = bool(motion_vae_target) and motion_vae_target.split('.')[-1].lower() == "vqvae"
        
        # Mean and std of the dataset
        # if True:
        if uses_vqvae:
            dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
            self.hparams.mean = np.load(pjoin(dis_data_root, "mean.npy"))
            self.hparams.std = np.load(pjoin(dis_data_root, "std.npy"))
        else:
            self.hparams.mean = np.load(pjoin(data_root, "Mean.npy"))
            self.hparams.std = np.load(pjoin(data_root, "Std.npy"))
        
        # Mean and std for fair evaluation
        if 'TMRMetrics' in cfg.METRIC.TYPE:
            self.hparams.mean_eval = np.load(pjoin(data_root, "tmr_mean.npy"))
            self.hparams.std_eval = np.load(pjoin(data_root, "tmr_std.npy"))
        else:
            dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "Comp_v6_KLD01", "meta")
            self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
            self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std.npy"))
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.HUMANML3D.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.HUMANML3D.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.HUMANML3D.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.HUMANML3D.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch - allow subclasses to override by checking if already set
        if not hasattr(self, 'DatasetEval') or self.DatasetEval is None:
            self.DatasetEval = Text2MotionDatasetEvalV3

        # Allow subclasses to override dataset selection by checking if Dataset is already set
        if not hasattr(self, 'Dataset') or self.Dataset is None:
            if cfg.TRAIN.STAGE == "vae":
                if uses_vqvae:
                    self.hparams.win_size = 64
                    self.Dataset = MotionDatasetVQ
                else:
                    # self.Dataset = MotionDataset
                    self.Dataset = Text2MotionDataset
            elif 'lm' in cfg.TRAIN.STAGE:
                self.hparams.code_path = cfg.DATASET.CODE_PATH
                self.hparams.task_path = cfg.DATASET.TASK_PATH
                self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT
                if uses_vqvae:
                    self.Dataset = Text2MotionDatasetCB
                else:
                    self.Dataset = Text2MotionDatasetCBV3
                    # self.DatasetEval = Text2MotionDatasetEvalV3
            elif cfg.TRAIN.STAGE == "token":
                self.Dataset = Text2MotionDatasetToken
                self.DatasetEval = Text2MotionDatasetToken
            elif cfg.TRAIN.STAGE == "m2t":
                self.Dataset = Text2MotionDatasetM2T
                self.DatasetEval = Text2MotionDatasetM2T
            else:
                self.Dataset = Text2MotionDataset
        else:
            # Dataset was pre-set by subclass, also set code_path if needed for lm stage
            if 'lm' in cfg.TRAIN.STAGE:
                self.hparams.code_path = cfg.DATASET.CODE_PATH
                self.hparams.task_path = cfg.DATASET.TASK_PATH
                self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT

        # Get additional info of the dataset
        # Skip sample_set loading if nfeats is already set by subclass (avoids double loading)
        if not hasattr(self, 'nfeats') or self.nfeats is None:
            self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
            # Only set nfeats if dataset has this attribute (not needed for token-based datasets)
            if hasattr(self._sample_set, 'nfeats'):
                self.nfeats = self._sample_set.nfeats
                cfg.DATASET.NFEATS = self.nfeats
            else:
                # Token-based datasets don't use nfeats, but set default for config compatibility
                self.nfeats = 263  # HumanML3D default feature dimension
                cfg.DATASET.NFEATS = self.nfeats
        else:
            # nfeats was pre-set by subclass, skip sample_set loading
            cfg.DATASET.NFEATS = self.nfeats
            self._sample_set = None
        
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def renorm4m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * eval_std + eval_mean
        features = (features - ori_mean) / ori_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
