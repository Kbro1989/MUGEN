"""SnapMoGen evaluator metrics (drop-in TM2TMetrics replacement).

Computes the official SnapMoGen protocol (snap-research/SnapMoGen,
``utils/eval_t2m.py``) inside the torchmetrics accumulate/compute pattern used
by TM2TMetrics, with identical metric key names so checkpoints / loggers /
``BestFIDCheckpoint`` work unchanged. Assigned to ``BaseMetrics.TM2TMetrics``
when the datamodule is snapmogen.

Protocol differences vs the HumanML3D TM2T evaluator:
- FID / Diversity use ``fid_emb`` (raw first-token encoder output);
  R-precision / Matching use the VAE mean ``mu`` -- two embedding caches each.
- Retrieval similarity is cosine; ``Matching_score`` is mean cosine similarity
  so HIGHER is better (opposite direction of the HML MMDist-style score).
- R-precision candidate pool ``R_size`` = 100 (official matching_pool_size),
  vs 32 in the HML protocol -- numbers are not comparable across datasets.
- Text goes in as raw caption strings (T5 inside the evaluator), not
  GloVe word_embs/pos_ohot.
"""

import logging
import os
from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric

from utils.snapmogen_evaluator import DEFAULT_EVALUATOR_DIR, load_snapmogen_evaluator

from .utils import (
    calculate_activation_statistics_np,
    calculate_diversity_np,
    calculate_frechet_distance_np,
    calculate_top_k,
)

# Same absl/rouge log-spam silencer as metrics/m2t.py (RougeScorer logs at INFO
# once per val epoch through the absl logger).
logging.getLogger("absl").setLevel(logging.WARNING)


def _cosine_sim_matrix(a: Tensor, b: Tensor) -> Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return a @ b.T


def _cosine_retrieval_block(all_texts: Tensor, all_motions: Tensor,
                            R_size: int, top_k: int):
    """Pooled cosine R-precision + matching over groups of R_size.

    Official SnapMoGen retrieval protocol: cosine similarity (HIGHER is
    closer), candidate pool of ``R_size`` (100). Shared by the T2M and M2T
    metric classes below.
    """
    count_seq = all_texts.shape[0]
    top_k_mat = torch.zeros((top_k, ))
    matching = 0.0
    for i in range(count_seq // R_size):
        group_texts = all_texts[i * R_size:(i + 1) * R_size]
        group_motions = all_motions[i * R_size:(i + 1) * R_size]
        sim_mat = _cosine_sim_matrix(group_texts, group_motions).nan_to_num()
        matching += sim_mat.trace()
        # Cosine: larger is closer -> sort descending.
        argsmax = torch.argsort(-sim_mat, dim=1)
        top_k_mat += calculate_top_k(argsmax, top_k=top_k).sum(axis=0)
    R_count = count_seq // R_size * R_size
    return matching, top_k_mat, R_count


class SnapMoGenTM2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='snapmogen',
                 top_k=3,
                 R_size=100,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 njoints=24,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.njoints = njoints
        self.name = "matching, fid, and diversity scores (SnapMoGen evaluator)"
        self.top_k = top_k
        self.R_size = int(cfg.METRIC.get('SNAPMOGEN_R_SIZE', R_size))
        self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task in ('t2m', 'gnu')
        self.diversity_times = diversity_times

        # Same external-MultiModality hook as TM2TMetrics (per-epoch monitor).
        self._extra_multimodality = None

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []
        if self.text:
            self.add_state("Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("gt_Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
            # CLIP Score (official SnapMoGen Table-3 metric, snap-research/
            # SnapMoGen ``evaluation_momask_plus``: ``matching_score_pred =
            # cosine_similarity_matrix(et, em_pred).trace() / nb_sample``).
            # That is the *same* mean text<->motion cosine similarity our
            # Matching_score computes -- the released code logs it as "Matching"
            # but the paper reports it as "CLIP Score". Surfaced here under the
            # paper's name so our reported numbers line up with Table 3.
            self.add_state("CLIP_Score", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("gt_CLIP_Score", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.Matching_metrics.extend(["CLIP_Score", "gt_CLIP_Score"])
            for k in range(1, top_k + 1):
                self.add_state(f"R_precision_top_{str(k)}", default=torch.tensor(0.0), dist_reduce_fx="sum")
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.add_state(f"gt_R_precision_top_{str(k)}", default=torch.tensor(0.0), dist_reduce_fx="sum")
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
            self.metrics.extend(self.Matching_metrics)

        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        self.add_state("Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gt_Diversity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # Cached batches. Retrieval (mu) and FID (fid_emb) live in different
        # embedding spaces under this evaluator, so cache both.
        self.add_state("text_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("recmotion_fid_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("gtmotion_fid_embeddings", default=[], dist_reduce_fx='cat')

        # Frozen SnapMoGen evaluator (motion tower in the checkpoint; T5 for the
        # text tower is lazy-loaded internally and never enters state_dict).
        self.evaluator = load_snapmogen_evaluator(
            device='cpu',
            ckpt_dir=cfg.METRIC.get('SNAPMOGEN_EVALUATOR_DIR', DEFAULT_EVALUATOR_DIR),
        )

    def _retrieval_block(self, all_texts: Tensor, all_motions: Tensor):
        """Pooled cosine R-precision + matching over groups of R_size."""
        assert all_texts.shape[0] > self.R_size
        return _cosine_retrieval_block(all_texts, all_motions, self.R_size, self.top_k)

    @torch.no_grad()
    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if sanity_flag:
            return metrics

        shuffle_idx = torch.randperm(count_seq)

        def gather(state):
            if type(state) == list:
                return torch.cat(state, axis=0).cpu().float()[shuffle_idx, :]
            return state.cpu().float()[shuffle_idx, :]

        all_genmotions = gather(self.recmotion_embeddings)
        all_gtmotions = gather(self.gtmotion_embeddings)
        all_genmotions_fid = gather(self.recmotion_fid_embeddings).numpy()
        all_gtmotions_fid = gather(self.gtmotion_fid_embeddings).numpy()

        if self.text:
            all_texts = gather(self.text_embeddings)
            matching, top_k_mat, R_count = self._retrieval_block(all_texts, all_genmotions)
            self.Matching_score += matching
            metrics["Matching_score"] = self.Matching_score / R_count
            # CLIP Score == this mean text<->motion cosine similarity (official
            # SnapMoGen Table-3 name). Aliased rather than recomputed so the two
            # are guaranteed identical.
            metrics["CLIP_Score"] = metrics["Matching_score"]
            for k in range(self.top_k):
                metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

            matching, top_k_mat, R_count = self._retrieval_block(all_texts, all_gtmotions)
            self.gt_Matching_score += matching
            metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
            metrics["gt_CLIP_Score"] = metrics["gt_Matching_score"]
            for k in range(self.top_k):
                metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # FID / Diversity on the fid_emb space (official protocol).
        mu, cov = calculate_activation_statistics_np(all_genmotions_fid)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions_fid)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions_fid, self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions_fid, self.diversity_times)

        if self._extra_multimodality is not None:
            metrics["MultiModality"] = self._extra_multimodality

        self.reset()
        self._extra_multimodality = None
        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor,
               feats_rst: Tensor,
               lengths_ref: List[int],
               lengths_rst: List[int],
               texts: List[str] = None,
               **kwargs):
        self.count += sum(lengths_ref)
        self.count_seq += len(lengths_ref)

        gt_fid, gt_mu = self.evaluator.encode_motion(
            feats_ref.float(), torch.as_tensor(lengths_ref))
        self.gtmotion_fid_embeddings.append(gt_fid.detach())
        self.gtmotion_embeddings.append(gt_mu.detach())

        rec_fid, rec_mu = self.evaluator.encode_motion(
            feats_rst.float(), torch.as_tensor(lengths_rst))
        self.recmotion_fid_embeddings.append(rec_fid.detach())
        self.recmotion_embeddings.append(rec_mu.detach())

        if self.text and texts is not None:
            text_emb = self.evaluator.encode_text(texts)
            self.text_embeddings.append(text_emb.detach())

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        """FID-space embedding -- the official MultiModality space."""
        fid_emb, _ = self.evaluator.encode_motion(
            feats.float(), torch.as_tensor(lengths))
        return fid_emb.detach()


class SnapMoGenM2TMetrics(Metric):
    """M2T (motion->text) metrics under the SnapMoGen evaluator.

    Drop-in ``M2TMetrics`` replacement for snapmogen (assigned to
    ``BaseMetrics.M2TMetrics``), with identical metric key names so the model's
    ``M2T_`` namespacing, the per-epoch console monitor and
    ``BestGNUCheckpoint`` all work unchanged.

    - NLG (``bleu_1`` / ``bleu_4`` / ``ROUGE_L`` / ``CIDEr`` + ``Bert_F1``) is
      dataset-agnostic and mirrors ``M2TMetrics`` (nlgmetricverse + bert_score,
      same ``skip_bert_score`` toggle).
    - M2T retrieval (``R_precision_top_k`` / ``Matching_score``) replaces the
      GloVe + TM2T towers with the SnapMoGen evaluator: predicted captions go
      through the T5 text tower (``mu``), GT motions through the motion tower
      (``mu``), cosine similarity over pools of ``R_size`` = 100 -- the same
      protocol as the T2M side, so Matching_score is mean cosine and HIGHER is
      better. The ``gt_*`` baseline encodes the first reference caption per
      sample (manual captions come first in ``all_captions``).
    """

    def __init__(self,
                 cfg,
                 dataname='snapmogen',
                 top_k=3,
                 R_size=100,
                 bleu_k=4,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 evaluator=None,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "matching, nlg scores (SnapMoGen evaluator)"
        self.top_k = top_k
        self.bleu_k = bleu_k
        self.R_size = int(cfg.METRIC.get('SNAPMOGEN_R_SIZE', R_size))
        self.diversity_times = diversity_times
        # Same toggle contract as M2TMetrics (set per-compute by the model:
        # skip on the per-epoch val pass, run at final test).
        self.skip_bert_score = False

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = []
        self.add_state("Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gt_Matching_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(f"R_precision_top_{str(k)}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(f"gt_R_precision_top_{str(k)}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
        self.metrics.extend(self.Matching_metrics)

        self.add_state("ROUGE_L", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("ROUGE_L")
        self.add_state("CIDEr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("CIDEr")

        # Cached batches (embeddings as states, raw strings as plain lists).
        self.pred_texts = []
        self.gt_texts = []
        self.add_state("predtext_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("gttext_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx='cat')

        # Reuse the (frozen) evaluator already loaded by SnapMoGenTM2TMetrics
        # when available -- shares the lazy T5 text tower instead of loading a
        # second ~1GB copy.
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            self.evaluator = load_snapmogen_evaluator(
                device='cpu',
                ckpt_dir=cfg.METRIC.get('SNAPMOGEN_EVALUATOR_DIR', DEFAULT_EVALUATOR_DIR),
            )

        if self.cfg.model.params.task in ('m2t', 'gnu'):
            from nlgmetricverse import NLGMetricverse, load_metric
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            self.nlg_evaluator = NLGMetricverse(metrics)

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor,
               pred_texts: List[str],
               gt_texts: List,
               lengths: List[int],
               word_embs: Tensor = None,
               pos_ohot: Tensor = None,
               text_lengths: Tensor = None):
        # word_embs / pos_ohot / text_lengths are HumanML3D GloVe inputs; the
        # SnapMoGen datamodule emits None for them and they are unused here
        # (kept in the signature so _update_m2t_metrics calls work unchanged).
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        pred_emb = self.evaluator.encode_text([t if t.strip() else '.' for t in pred_texts])
        self.predtext_embeddings.append(pred_emb.detach())

        _, gt_mu = self.evaluator.encode_motion(
            feats_ref.float(), torch.as_tensor(lengths))
        self.gtmotion_embeddings.append(gt_mu.detach())

        gt_first = [
            refs[0] if isinstance(refs, (list, tuple)) and len(refs) else str(refs)
            for refs in gt_texts
        ]
        gttext_emb = self.evaluator.encode_text(gt_first)
        self.gttext_embeddings.append(gttext_emb.detach())

        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)

    @torch.no_grad()
    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if sanity_flag:
            return metrics

        shuffle_idx = torch.randperm(count_seq)

        def gather(state):
            if type(state) == list:
                return torch.cat(state, axis=0).cpu().float()[shuffle_idx, :]
            return state.cpu().float()[shuffle_idx, :]

        all_predtexts = gather(self.predtext_embeddings)
        all_gttexts = gather(self.gttext_embeddings)
        all_motions = gather(self.gtmotion_embeddings)

        # Retrieval (skipped when the pool is too small, e.g. DEBUG runs).
        if count_seq >= self.R_size:
            matching, top_k_mat, R_count = _cosine_retrieval_block(
                all_predtexts, all_motions, self.R_size, self.top_k)
            self.Matching_score += matching
            metrics["Matching_score"] = self.Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

            matching, top_k_mat, R_count = _cosine_retrieval_block(
                all_gttexts, all_motions, self.R_size, self.top_k)
            self.gt_Matching_score += matching
            metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # NLG metrics (same flow / fallbacks as M2TMetrics.compute).
        print(f"Computing NLG metrics for {len(self.pred_texts)} samples...")
        try:
            scores = self.nlg_evaluator(predictions=self.pred_texts,
                                        references=self.gt_texts)
            for key in scores.keys():
                if 'bleu' in key:
                    metrics[key] = torch.tensor(scores[key]['score'], device=self.device)

            if "rouge" in scores:
                rouge_data = scores["rouge"]
                if isinstance(rouge_data, dict):
                    if "rougeL" in rouge_data:
                        rouge_l_val = rouge_data["rougeL"]
                    elif "score" in rouge_data:
                        rouge_l_val = rouge_data["score"]
                    else:
                        rouge_l_val = rouge_data.get("rouge-l", rouge_data.get("rougeLsum", 0.0))
                        if isinstance(rouge_l_val, dict):
                            rouge_l_val = rouge_l_val.get("score", rouge_l_val.get("fmeasure", 0.0))
                else:
                    rouge_l_val = float(rouge_data) if rouge_data else 0.0
                metrics["ROUGE_L"] = torch.tensor(rouge_l_val, device=self.device)
            else:
                metrics["ROUGE_L"] = torch.tensor(0.0, device=self.device)

            if "cider" in scores:
                cider_data = scores["cider"]
                if isinstance(cider_data, dict) and "score" in cider_data:
                    cider_val = cider_data["score"]
                else:
                    cider_val = float(cider_data) if cider_data else 0.0
                metrics["CIDEr"] = torch.tensor(cider_val, device=self.device)
            else:
                metrics["CIDEr"] = torch.tensor(0.0, device=self.device)
        except Exception as e:
            print(f"Warning: NLG metrics computation failed: {e}")
            metrics["bleu_1"] = torch.tensor(0.0, device=self.device)
            metrics["bleu_4"] = torch.tensor(0.0, device=self.device)
            metrics["ROUGE_L"] = torch.tensor(0.0, device=self.device)
            metrics["CIDEr"] = torch.tensor(0.0, device=self.device)

        skip_bert_score = (self.skip_bert_score
                           or os.environ.get('SKIP_BERT_SCORE', '0') == '1')
        if skip_bert_score:
            metrics["Bert_F1"] = torch.tensor(0.0, device=self.device)
        else:
            try:
                from bert_score import score as score_bert
                max_samples = min(len(self.pred_texts), 500)
                # nthreads=0: serial IDF dict, no multiprocessing.Pool fork --
                # forked Pool workers inherit Lightning's swallow-SIGTERM
                # handler and hang Pool teardown (see metrics/m2t.py).
                P, R, F1 = score_bert(self.pred_texts[:max_samples],
                                      self.gt_texts[:max_samples],
                                      lang='en',
                                      rescale_with_baseline=True,
                                      idf=True,
                                      device=self.device,
                                      verbose=False,
                                      nthreads=0)
                metrics["Bert_F1"] = F1.mean().to(self.device)
            except Exception as e:
                print(f"Warning: BERTScore computation failed: {e}")
                metrics["Bert_F1"] = torch.tensor(0.0, device=self.device)

        self.reset()
        self.pred_texts = []
        self.gt_texts = []
        return {**metrics}
