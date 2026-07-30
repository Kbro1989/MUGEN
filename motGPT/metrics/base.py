from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2t import M2TMetrics
from .m2m import PredMetrics
from .tmr import TMRMetrics
from .token_consistency import TokenConsistencyMetrics


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints
        # cfg.METRIC.TM2T.t2m_textencoder.params['dataset'] = datamodule.name

        data_name = datamodule.name
        print('in BaseMetrics, data_name:', data_name)
        if data_name == 'snapmogen':
            # SnapMoGen has no TM2T/GloVe evaluator; the official SnapMoGen
            # evaluator (TMR-style, cosine retrieval, pool 100) replaces it.
            # Kept under the TM2TMetrics attribute name so model code and
            # checkpoint callbacks ('Metrics/FID' etc.) work unchanged.
            from .snapmogen import SnapMoGenM2TMetrics, SnapMoGenTM2TMetrics
            if 'TM2TMetrics' in cfg.METRIC.TYPE:
                self.TM2TMetrics = SnapMoGenTM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    njoints=njoints,
                )
            if 'M2TMetrics' in cfg.METRIC.TYPE:
                # GNU understanding branch under the SnapMoGen evaluator; share
                # the frozen evaluator (and its lazy T5) with the T2M metric.
                self.M2TMetrics = SnapMoGenM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    evaluator=(self.TM2TMetrics.evaluator
                               if hasattr(self, 'TM2TMetrics') else None),
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
        elif data_name in ["humanml3d", "kit", 'motionx', 'tomato']:
            if 'TM2TMetrics' in cfg.METRIC.TYPE:
                self.TM2TMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    njoints=njoints,
                )
            if 'M2TMetrics' in cfg.METRIC.TYPE:
                self.M2TMetrics = M2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    w_vectorizer=datamodule.hparams.w_vectorizer,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
            if 'MMMetrics' in cfg.METRIC.TYPE:
                self.MMMetrics = MMMetrics(
                cfg=cfg,
                dataname=data_name,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                njoints=njoints,
            )
            if 'TemosMetric' in cfg.METRIC.TYPE:
                from .compute import ComputeMetrics
                self.TemosMetric = ComputeMetrics(
                    njoints=njoints,
                    jointstype=cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            if 'TMRMetrics' in cfg.METRIC.TYPE:
                self.TMRMetrics = TMRMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    threshold_selfsim_metrics=0.95
                )

        if 'MRMetrics' in cfg.METRIC.TYPE:
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if 'PredMetrics' in cfg.METRIC.TYPE:
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )

        if 'TokenConsistencyMetrics' in cfg.METRIC.TYPE:
            self.TokenConsistencyMetrics = TokenConsistencyMetrics(
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
