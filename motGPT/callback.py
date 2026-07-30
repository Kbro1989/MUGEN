import os
import csv
import numpy as np
from typing import Optional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


class LossCSVLogger(Callback):
    """
    Callback to log total loss and orthogonality loss at each epoch end
    and save to a CSV file when training completes.
    """

    CSV_FIELDS = [
        "epoch", "total_loss", "gpt_ce_loss", "ortho_loss", "res_loss",
        "pred_rec_loss", "decoder_only_loss", "residual_pred_norm", "reskd_loss",
        # AR-mode validation metrics aligned with the Stage-3 joint path.
        "val_ar_joint_rec", "val_ar_codeword_only_loss", "val_ar_residual_pred_norm",
        "val_ar_residual_kd_loss", "val_ar_token_match_rate",
    ]

    def __init__(self, output_dir: str, filename: str = "epoch_losses.csv"):
        """
        Args:
            output_dir: Directory where the CSV file will be saved
            filename: Name of the CSV file (default: epoch_losses.csv)
        """
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.epoch_losses = []  # list of dicts keyed by CSV_FIELDS

    def _get_metric(self, metrics, keys, default: Optional[float] = 0.0):
        """Helper to extract a metric value from callback_metrics by trying multiple keys."""
        for key in keys:
            if key in metrics:
                val = metrics[key]
                return val.item() if hasattr(val, 'item') else float(val)
        return default

    def _write_csv(self):
        """Rewrite the full CSV with all accumulated epoch rows."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, self.filename)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for row in self.epoch_losses:
                writer.writerow({k: row.get(k, "") for k in self.CSV_FIELDS})

    def _current_row(self, epoch):
        """Return the row dict for `epoch`, creating it if needed."""
        for row in self.epoch_losses:
            if row["epoch"] == epoch:
                return row
        row = {"epoch": epoch}
        self.epoch_losses.append(row)
        return row

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record training-side losses at the end of each training epoch."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        total_loss = self._get_metric(metrics, ["train/loss_epoch", "train/loss", "total/train"], default=None)
        ortho_loss = self._get_metric(metrics, ["train/ortho_loss_epoch", "train/ortho_loss"])
        res_loss = self._get_metric(metrics, ["train/res_loss_avg", "train/res_loss"])
        pred_rec_loss = self._get_metric(metrics, ["train/pred_lat_rec_epoch", "train/pred_lat_rec"])
        gpt_ce_loss = self._get_metric(metrics, ["train/gpt_ce_loss_epoch", "train/gpt_ce_loss"])
        decoder_only_loss = self._get_metric(metrics, ["train/decoder_only_loss_epoch", "train/decoder_only_loss"])
        residual_pred_norm = self._get_metric(metrics, ["train/residual_pred_norm_epoch", "train/residual_pred_norm"])
        reskd_loss = self._get_metric(metrics, ["train/reskd_loss_epoch", "train/reskd_loss", "reskd/loss/train"])

        row = self._current_row(epoch)
        row.update({
            "total_loss": total_loss,
            "gpt_ce_loss": gpt_ce_loss,
            "ortho_loss": ortho_loss,
            "res_loss": res_loss,
            "pred_rec_loss": pred_rec_loss,
            "decoder_only_loss": decoder_only_loss,
            "residual_pred_norm": residual_pred_norm,
            "reskd_loss": reskd_loss,
        })

        # Also log to console for visibility
        if trainer.global_rank == 0:
            parts = [f"total_loss={total_loss}"]
            if gpt_ce_loss:
                parts.append(f"gpt_ce_loss={gpt_ce_loss:.6f}")
            if ortho_loss:
                parts.append(f"ortho_loss={ortho_loss}")
            if res_loss:
                parts.append(f"res_loss={res_loss}")
            if pred_rec_loss:
                parts.append(f"PredRecLoss={pred_rec_loss:.6f}")
            if decoder_only_loss:
                parts.append(f"DecoderOnlyLoss={decoder_only_loss:.6f}")
            if residual_pred_norm:
                parts.append(f"ResPredNorm={residual_pred_norm:.6f}")
            if reskd_loss:
                parts.append(f"ResKDLoss={reskd_loss:.6f}")
            print(f"[LossCSVLogger] Epoch {epoch}: {', '.join(parts)}")
            # Refresh CSV on rank 0 each epoch
            self._write_csv()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record AR-mode validation losses and rewrite CSV.

        Uses ``on_validation_end`` (not ``on_validation_epoch_end``) because in
        Lightning 2.x callbacks' ``on_validation_epoch_end`` fires *before* the
        LightningModule's, so ``callback_metrics`` would still hold stale
        values from the previous validation cycle.
        """
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        val_ar_joint_rec = self._get_metric(
            metrics, ["val/ar_joint_rec_epoch", "val/ar_joint_rec"], default=None)
        val_ar_codeword_only_loss = self._get_metric(
            metrics, ["val/ar_codeword_only_loss_epoch", "val/ar_codeword_only_loss"], default=None)
        val_ar_residual_pred_norm = self._get_metric(
            metrics, ["val/ar_residual_pred_norm_epoch", "val/ar_residual_pred_norm"], default=None)
        val_ar_residual_kd_loss = self._get_metric(
            metrics, ["val/ar_residual_kd_loss_epoch", "val/ar_residual_kd_loss"], default=None)
        val_ar_token_match_rate = self._get_metric(
            metrics, ["val/ar_token_match_rate_epoch", "val/ar_token_match_rate"], default=None)

        # Skip if no val metrics were logged (e.g. sanity check w/o t2m val)
        if (val_ar_joint_rec is None and val_ar_codeword_only_loss is None
                and val_ar_residual_pred_norm is None and val_ar_residual_kd_loss is None
                and val_ar_token_match_rate is None):
            return

        row = self._current_row(epoch)
        if val_ar_joint_rec is not None:
            row["val_ar_joint_rec"] = val_ar_joint_rec
        if val_ar_codeword_only_loss is not None:
            row["val_ar_codeword_only_loss"] = val_ar_codeword_only_loss
        if val_ar_residual_pred_norm is not None:
            row["val_ar_residual_pred_norm"] = val_ar_residual_pred_norm
        if val_ar_residual_kd_loss is not None:
            row["val_ar_residual_kd_loss"] = val_ar_residual_kd_loss
        if val_ar_token_match_rate is not None:
            row["val_ar_token_match_rate"] = val_ar_token_match_rate

        parts = []
        if val_ar_joint_rec is not None:
            parts.append(f"val_ARJointRec={val_ar_joint_rec:.6f}")
        if val_ar_codeword_only_loss is not None:
            parts.append(f"val_ARCodewordOnly={val_ar_codeword_only_loss:.6f}")
        if val_ar_residual_pred_norm is not None:
            parts.append(f"val_ARResPredNorm={val_ar_residual_pred_norm:.6f}")
        if val_ar_residual_kd_loss is not None:
            parts.append(f"val_ARResKDLoss={val_ar_residual_kd_loss:.6f}")
        if val_ar_token_match_rate is not None:
            parts.append(f"val_ARTokenMatch={val_ar_token_match_rate:.4f}")
        if parts:
            print(f"[LossCSVLogger] Epoch {epoch} (val): {', '.join(parts)}")

        self._write_csv()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Final CSV flush at the end of training."""
        if trainer.global_rank != 0:
            return
        self._write_csv()
        csv_path = os.path.join(self.output_dir, self.filename)
        print(f"[LossCSVLogger] Saved epoch losses to: {csv_path}")


class BestFIDCheckpoint(Callback):
    """Save best_fid.ckpt: checkpoint with the lowest validation FID seen so far.

    Uses the standard resume-safety pattern (state_dict / load_state_dict) so the
    best FID and epoch persist across resumes.
    """

    FID_KEY = "Metrics/FID"

    def __init__(self, save_dir: str, logger=None, filename: str = "best_fid.ckpt"):
        super().__init__()
        self.save_dir = save_dir
        self.logger = logger
        self.filename = filename
        self.best_fid = float("inf")
        self.best_epoch = -1

    def _get(self, metrics, key):
        if key in metrics:
            v = metrics[key]
            return v.item() if hasattr(v, "item") else float(v)
        return None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        if trainer.sanity_checking:
            return

        fid = self._get(trainer.callback_metrics, self.FID_KEY)
        if fid is None:
            return

        if fid < self.best_fid:
            prev = self.best_fid
            self.best_fid = fid
            self.best_epoch = trainer.current_epoch
            save_path = os.path.join(self.save_dir, "checkpoints", self.filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_checkpoint(save_path)
            msg = (f"[BestFID] Epoch {self.best_epoch}: new best FID={fid:.4f} "
                   f"(prev={prev:.4f}) -> {save_path}")
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        msg = f"[BestFID] Training finished. Best epoch: {self.best_epoch} (FID={self.best_fid:.4f})"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def state_dict(self) -> dict:
        return {"best_fid": self.best_fid, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_fid = state_dict.get("best_fid", float("inf"))
        self.best_epoch = state_dict.get("best_epoch", -1)
        msg = (f"[BestFID] Resumed state: best_epoch={self.best_epoch} "
               f"best_fid={self.best_fid:.4f}")
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


class BestGNUCheckpoint(Callback):
    """Save best_gnu.ckpt: the checkpoint with the best *joint* G+U score.

    ``best``/``best_fid`` are driven entirely by T2M (generation) quality, so for
    the GNU phase they systematically ship a model whose M2T (understanding)
    quality is far past its peak. This callback scores BOTH directions every
    validation epoch and keeps the best joint model.

    Scoring uses the standard benchmark "Average" of each direction (the figure
    the user referenced), then a balance-enforcing geometric mean.

      Understanding (the exact M2T Average, R-Precision included):
        R_u   = (R1 + R2 + R3) / 3                  # M2T retrieval R-precision
        Avg_U = mean(100*R_u, 100*B1, 100*B4, 100*RL, 100*C [, 100*S])
                # BERTScore S is only folded in when actually computed (off on
                # val by default -> 5-term mean; on at test -> 6-term mean).

      Generation (v2, 2026-07-03; every term -> higher-is-better, ~0-100;
      MMDist handled so LOWER is BETTER and *beating* GT is rewarded, fixing the
      degenerate direction where the model's MMDist drops below MMDist_gt):
        R_score      = 100 * (R1 + R2 + R3) / 3
        FID_score    = 100 * fid_tau / (fid_tau + FID)  # rational map: score 50
                       # at FID == fid_tau, NO saturation. (The v1 exp map
                       # 100*exp(-FID/tau) was flat over SnapMoGen's achievable
                       # FID 15-50 at tau=10 -> inside an arithmetic mean the
                       # R/Match/Div terms could outvote FID entirely.)
        MMDist_score = 100 * (MMDist_gt / MMDist)   # lower MMDist -> higher; =100 at GT, >100 if better
        Div_score    = 100 * max(0, 1 - |Div - Div_gt| / Div_gt)
        Avg_G        = weighted GEOMETRIC mean of the four terms
                       (weights R:FID:Match:Div default 2:2:1:1,
                        cfg METRIC.GNU_G_WEIGHTS)

      The v1 arithmetic mean allowed compensation: a checkpoint could buy score
      with R/Match/Div while FID stagnated. The geometric mean multiplies
      relative losses, so the selected checkpoint must have R-precision AND FID
      near their bests SIMULTANEOUSLY -- neither can be traded for the other.
      Div stays in as a variance-collapse guard (weight 1).

      Joint:
        S = Avg_G**alpha * Avg_U**(1-alpha)         # geometric mean (alpha=0.5)

    Geometric mean (vs. a weighted sum) means tanking either direction tanks the
    joint score, so the saved model must be good at BOTH. ``alpha`` shifts the
    balance: >0.5 favours generation, <0.5 favours understanding.

    ``SCORE_VERSION`` guards resumes: a checkpoint trained under the v1 formula
    carries a stale ``best_score`` that is not comparable to v2 values, so a
    version mismatch resets the incumbent and lets the new formula re-select.
    """

    SCORE_VERSION = 2  # v2: geometric-mean G side + rational FID map

    T2M = {
        "fid": "Metrics/FID",
        "r1": "Metrics/R_precision_top_1",
        "r2": "Metrics/R_precision_top_2",
        "r3": "Metrics/R_precision_top_3",
        "mm": "Metrics/Matching_score",
        "mm_gt": "Metrics/gt_Matching_score",
        "div": "Metrics/Diversity",
    }
    M2T = {
        "r1": "Metrics/M2T_R_precision_top_1",
        "r2": "Metrics/M2T_R_precision_top_2",
        "r3": "Metrics/M2T_R_precision_top_3",
        "b1": "Metrics/M2T_bleu_1",
        "b4": "Metrics/M2T_bleu_4",
        "rl": "Metrics/M2T_ROUGE_L",
        "cider": "Metrics/M2T_CIDEr",
        "bert": "Metrics/M2T_Bert_F1",
    }

    def __init__(self, save_dir: str, div_gt: float = 9.5, fid_tau: float = 1.0,
                 alpha: float = 0.5, logger=None, filename: str = "best_gnu.ckpt",
                 matching_higher_is_better: bool = False, g_weights=None):
        super().__init__()
        self.save_dir = save_dir
        self.div_gt = float(div_gt)
        self.fid_tau = float(fid_tau)
        self.alpha = float(alpha)
        self.logger = logger
        self.filename = filename
        # HumanML3D's Matching_score is an L2 distance (MMDist, LOWER is
        # better); the SnapMoGen evaluator's is mean cosine similarity (HIGHER
        # is better). True flips the MMDist_score normalization direction.
        self.matching_higher_is_better = bool(matching_higher_is_better)
        # Geometric-mean weights over (R, FID, Match, Div). R and FID are the
        # headline selection targets; Match/Div act as guards.
        if g_weights is None:
            g_weights = (2.0, 2.0, 1.0, 1.0)
        self.g_weights = tuple(float(w) for w in g_weights)
        if len(self.g_weights) != 4 or any(w < 0 for w in self.g_weights) \
                or sum(self.g_weights) <= 0:
            raise ValueError(
                f"g_weights must be 4 non-negative floats (R, FID, Match, Div) "
                f"with a positive sum, got {g_weights}")
        self.best_score = float("-inf")
        self.best_epoch = -1

    def _get(self, m, key):
        if key in m:
            v = m[key]
            return v.item() if hasattr(v, "item") else float(v)
        return None

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _avg_u(self, m):
        """M2T benchmark Average (R-Precision + BLEU/ROUGE/CIDEr [+ BERTScore])."""
        r = [self._get(m, self.M2T[k]) for k in ("r1", "r2", "r3")]
        b1 = self._get(m, self.M2T["b1"])
        b4 = self._get(m, self.M2T["b4"])
        rl = self._get(m, self.M2T["rl"])
        cider = self._get(m, self.M2T["cider"])
        if any(v is None for v in r) or None in (b1, b4, rl, cider):
            return None
        r_bar = sum(r) / 3.0
        terms = [100.0 * r_bar, 100.0 * b1, 100.0 * b4, 100.0 * rl, 100.0 * cider]
        bert = self._get(m, self.M2T["bert"])
        if bert is not None and bert > 0.0:  # off on val -> 0/missing; only test
            terms.append(100.0 * bert)
        return sum(terms) / len(terms)

    def _avg_g(self, m):
        """T2M score: weighted geometric mean of higher-is-better terms (~0-100).

        v2 (2026-07-03): rational FID map + geometric mean, so R-precision and
        FID must be near their bests SIMULTANEOUSLY (no term can buy score for
        another; see class docstring).
        """
        r = [self._get(m, self.T2M[k]) for k in ("r1", "r2", "r3")]
        fid = self._get(m, self.T2M["fid"])
        mm = self._get(m, self.T2M["mm"])
        mm_gt = self._get(m, self.T2M["mm_gt"])
        div_ = self._get(m, self.T2M["div"])
        if any(v is None for v in r) or None in (fid, mm, div_):
            return None
        r_score = 100.0 * (sum(r) / 3.0)
        # Rational map: 100 at FID 0, 50 at FID == fid_tau, no saturation --
        # constant RELATIVE sensitivity d(log score) = -dFID/(fid_tau+FID)
        # across the whole achievable range (the v1 exp map was flat for
        # FID >> tau, silencing FID inside the average).
        fid_score = 100.0 * self.fid_tau / (self.fid_tau + max(fid, 0.0))
        if self.matching_higher_is_better:
            # SnapMoGen cosine Matching_score: higher is better. =100 at GT,
            # >100 when the model beats the GT captions' alignment.
            if mm_gt is not None and mm_gt > 1e-8:
                mm_score = 100.0 * (mm / mm_gt)
            else:
                mm_score = 100.0 * max(mm, 0.0)  # no-GT fallback
        # MMDist: lower is better. Normalize to GT so it is 100 at GT and >100
        # when the model beats GT (the case the user flagged) -- never penalized
        # for dropping below MMDist_gt.
        elif mm_gt is not None and mm > 1e-8:
            mm_score = 100.0 * (mm_gt / mm)
        else:
            mm_score = 100.0 * float(np.exp(-max(mm, 0.0)))  # no-GT fallback
        div_dev = abs(div_ - self.div_gt) / max(self.div_gt, 1e-8)
        div_score = 100.0 * max(0.0, 1.0 - div_dev)
        terms = (r_score, fid_score, mm_score, div_score)
        w_sum = sum(self.g_weights)
        log_mean = sum(
            w * float(np.log(max(t, 1e-6)))
            for w, t in zip(self.g_weights, terms)
        ) / w_sum
        return float(np.exp(log_mean))

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # on_validation_end (not _epoch_end) so the LM-hook-logged metrics are
        # already in callback_metrics (the LM logs them in its on_validation_epoch_end,
        # which runs before this on_validation_end -- not the callback _epoch_end hook).
        if trainer.global_rank != 0 or trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        avg_g = self._avg_g(m)
        avg_u = self._avg_u(m)
        if avg_g is None or avg_u is None:
            return
        g = max(avg_g, 1e-6)
        u = max(avg_u, 1e-6)
        score = float(g ** self.alpha * u ** (1.0 - self.alpha))
        epoch = trainer.current_epoch
        if score > self.best_score:
            prev = self.best_score
            self.best_score = score
            self.best_epoch = epoch
            save_path = os.path.join(self.save_dir, "checkpoints", self.filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_checkpoint(save_path)
            fid = self._get(m, self.T2M["fid"])
            r1 = self._get(m, self.T2M["r1"])
            self._log(f"[BestGNU] Epoch {epoch}: new best joint={score:.3f} "
                      f"(prev={prev:.3f}) Avg_G={avg_g:.2f} Avg_U={avg_u:.2f} "
                      f"[FID={fid:.2f} R@1={r1:.3f}] -> {save_path}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        self._log(f"[BestGNU] Training finished. Best epoch: {self.best_epoch} "
                  f"(joint={self.best_score:.3f})")

    def state_dict(self) -> dict:
        return {"best_score": self.best_score, "best_epoch": self.best_epoch,
                "score_version": self.SCORE_VERSION}

    def load_state_dict(self, state_dict: dict) -> None:
        version = state_dict.get("score_version", 1)
        if version != self.SCORE_VERSION:
            # Scores from another formula version are not comparable; keeping
            # the stale incumbent would block every future save (v1 arithmetic
            # scores run higher than v2 geometric ones). Reset and re-select.
            self._log(f"[BestGNU] Score formula changed (v{version} -> "
                      f"v{self.SCORE_VERSION}); resetting best "
                      f"(was epoch={state_dict.get('best_epoch', -1)}, "
                      f"joint={state_dict.get('best_score', float('-inf')):.3f}) "
                      f"so selection restarts under the new score.")
            self.best_score = float("-inf")
            self.best_epoch = -1
            return
        self.best_score = state_dict.get("best_score", float("-inf"))
        self.best_epoch = state_dict.get("best_epoch", -1)
        self._log(f"[BestGNU] Resumed: best_epoch={self.best_epoch} "
                  f"best_joint={self.best_score:.3f}")


class T2MVisualizationCallback(Callback):
    """Render the full T2M test set (GT + generated) into per-sample folders.

    Output layout under `output_dir`:
        <keyid>/
            caption.txt
            gt.mp4
            pred.mp4

    Rendering is dispatched to a multiprocessing pool (spawn context) so the
    GPU-side test loop is not blocked. Existing complete pairs are skipped,
    making re-runs idempotent / resumable.
    """

    def __init__(self, output_dir, fps: int = 20, num_workers: Optional[int] = None,
                 replication_only_first: bool = True, num_samples: Optional[int] = None,
                 seed: int = 0, logger=None, sample: bool = True,
                 temperature: Optional[float] = None,
                 render_video: bool = True, save_feats: bool = False):
        """
        Args:
            num_samples: If None, render the full test set. Otherwise render
                only this many randomly chosen samples (deterministic given
                `seed`).
            sample: If True (default), the generated motion is drawn from the
                calibrated conditional sampler (the honest generative protocol,
                same as the FID pass) instead of the deterministic mu-decode.
            temperature: Sampling temperature; None falls back to the model's
                ``eval_sample_temperature``.
        """
        super().__init__()
        self.output_dir = str(output_dir)
        self.fps = int(fps)
        if num_workers is None:
            num_workers = max(1, min(16, (os.cpu_count() or 4) // 2))
        self.num_workers = int(num_workers)
        self.replication_only_first = bool(replication_only_first)
        self.num_samples = None if num_samples is None else int(num_samples)
        self.seed = int(seed)
        self.logger = logger
        self.sample = bool(sample)
        self.temperature = temperature
        # npy-only mode (mp4 disabled) plus evaluation-feature dumping.
        self.render_video = bool(render_video)
        self.save_feats = bool(save_feats)
        self._executor = None
        self._futures = []
        self._invocation_count = 0
        self._submitted = 0
        self._skipped_existing = 0
        self._completed = 0
        self._probed = False
        self._warned_no_t2m = False
        self._global_sample_idx = 0
        self._chosen_indices = None  # set[int] | None  (None = render all)

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        self._invocation_count += 1
        if self.replication_only_first and self._invocation_count > 1:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        # Use spawn to avoid forking a CUDA-initialized parent process.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        from motGPT.utils.render_utils import _silence_worker_warnings
        ctx = _mp.get_context('spawn')
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_silence_worker_warnings,
        )
        self._futures = []
        self._submitted = 0
        self._skipped_existing = 0
        self._completed = 0
        self._probed = False
        self._global_sample_idx = 0
        self._chosen_indices = None

        # If subsampling is requested, pre-pick which global sample indices to
        # render (deterministic given `self.seed`).
        if self.num_samples is not None:
            total = None
            try:
                dl = trainer.test_dataloaders
                if isinstance(dl, (list, tuple)):
                    dl = dl[0]
                total = len(dl.dataset)
            except Exception:
                try:
                    total = len(trainer.datamodule.test_dataset)
                except Exception:
                    total = None
            if total is None:
                # Fallback: just render the first N samples in order.
                self._chosen_indices = set(range(self.num_samples))
                self._log(f"[T2MVis] Test set size unknown; rendering first "
                          f"{self.num_samples} samples in order.")
            else:
                import random as _random
                rng = _random.Random(self.seed)
                k = min(self.num_samples, total)
                self._chosen_indices = set(rng.sample(range(total), k))
                self._log(f"[T2MVis] Subsampling {k}/{total} samples "
                          f"(seed={self.seed}).")

        self._log(f"[T2MVis] Visualization enabled -> {self.output_dir} "
                  f"(workers={self.num_workers}, fps={self.fps})")

    def _resolve_keyid(self, batch, i: int, batch_idx: int) -> str:
        fnames = batch.get('fname', None)
        if fnames is not None and i < len(fnames) and fnames[i]:
            base = os.path.basename(str(fnames[i]))
            keyid, _ = os.path.splitext(base)
            if keyid:
                return keyid
        return f"b{batch_idx:05d}_i{i:03d}"

    def _resolve_caption(self, batch, i: int) -> str:
        texts = batch.get('text', None)
        if texts is None or i >= len(texts):
            return ""
        t = texts[i]
        if isinstance(t, (list, tuple)):
            return "\n".join(str(x) for x in t)
        return str(t)

    @staticmethod
    def _has_pair(sample_dir: str) -> bool:
        return (os.path.exists(os.path.join(sample_dir, 'gt.mp4'))
                and os.path.exists(os.path.join(sample_dir, 'pred.mp4')))

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if trainer.global_rank != 0:
            return
        if self._executor is None:
            return  # Disabled (e.g. replication_only_first and not the first invocation).
        # Skip multimodality repeats.
        try:
            if getattr(pl_module.trainer.datamodule, 'is_mm', False):
                return
        except Exception:
            pass
        # Need val_t2m_forward + a t2m task.
        if not hasattr(pl_module, 'val_t2m_forward'):
            if not self._warned_no_t2m:
                self._log("[T2MVis] pl_module has no val_t2m_forward; skipping.")
                self._warned_no_t2m = True
            return
        # GNU joint models still expose val_t2m_forward (the generation branch),
        # so visualize their T2M output too -- only skip genuinely non-T2M tasks.
        task = getattr(getattr(pl_module, 'hparams', None), 'task', None)
        if task is not None and task not in ('t2m', 'gnu'):
            if not self._warned_no_t2m:
                self._log(f"[T2MVis] task={task!r} not in ('t2m', 'gnu'); skipping.")
                self._warned_no_t2m = True
            return

        import torch as _torch
        from motGPT.utils.render_utils import render_fast_to_file

        # Figure out batch size + which global indices fall in this batch so
        # we can short-circuit when subsampling and no chosen sample is here.
        try:
            B = int(len(batch.get('length', batch.get('text', []))))
        except Exception:
            B = 0
        if B == 0:
            try:
                B = int(batch['motion'].shape[0])
            except Exception:
                B = 0
        batch_base = self._global_sample_idx
        if self._chosen_indices is not None:
            keep_mask = [(batch_base + i) in self._chosen_indices for i in range(B)]
            if not any(keep_mask):
                self._global_sample_idx += B
                return
        else:
            keep_mask = [True] * B

        with _torch.no_grad():
            try:
                rs_set = pl_module.val_t2m_forward(
                    batch, sample=self.sample, temperature=self.temperature)
            except Exception as exc:
                self._log(f"[T2MVis] val_t2m_forward failed on batch {batch_idx}: {exc}")
                self._global_sample_idx += B
                return

        joints_ref = rs_set.get('joints_ref', None)
        joints_rst = rs_set.get('joints_rst', None)
        lengths = rs_set.get('length', None)
        # Evaluation features after renorm4t2m: the exact tensor fed to
        # encode_motion, so the normalisation convention is correct by construction.
        feats_ref = rs_set.get('m_ref', None) if self.save_feats else None
        feats_rst = rs_set.get('m_rst', None) if self.save_feats else None
        if joints_ref is None or joints_rst is None or lengths is None:
            self._global_sample_idx += B
            return

        B = int(joints_ref.shape[0])
        for i in range(B):
            if i < len(keep_mask) and not keep_mask[i]:
                continue
            keyid = self._resolve_keyid(batch, i, batch_idx)
            sample_dir = os.path.join(self.output_dir, keyid)
            os.makedirs(sample_dir, exist_ok=True)

            # Always (re)write caption — cheap and ensures up-to-date text.
            try:
                with open(os.path.join(sample_dir, 'caption.txt'), 'w', encoding='utf-8') as f:
                    f.write(self._resolve_caption(batch, i))
            except Exception as exc:
                self._log(f"[T2MVis] failed to write caption for {keyid}: {exc}")

            L = max(1, int(lengths[i]))
            ref_np = joints_ref[i, :L].detach().cpu().numpy()
            rst_np = joints_rst[i, :L].detach().cpu().numpy()

            # Also persist the raw 3D joint coordinates (the source arrays for
            # higher-quality re-rendering later). Shape [T, J, 3] with J=22
            # (HumanML3D) or 24 (SnapMoGen). Written even when the mp4 pair
            # already exists; the existence guards keep re-runs idempotent.
            gt_npy = os.path.join(sample_dir, 'gt.npy')
            pred_npy = os.path.join(sample_dir, 'pred.npy')
            try:
                if not os.path.exists(gt_npy):
                    np.save(gt_npy, ref_np)
                if not os.path.exists(pred_npy):
                    np.save(pred_npy, rst_np)
            except Exception as exc:
                self._log(f"[T2MVis] failed to save joints for {keyid}: {exc}")

            if feats_ref is not None and feats_rst is not None:
                gt_feat = os.path.join(sample_dir, 'gt_feat.npy')
                pred_feat = os.path.join(sample_dir, 'pred_feat.npy')
                try:
                    if not os.path.exists(gt_feat):
                        np.save(gt_feat, feats_ref[i, :L].detach().cpu().numpy())
                    if not os.path.exists(pred_feat):
                        np.save(pred_feat, feats_rst[i, :L].detach().cpu().numpy())
                except Exception as exc:
                    self._log(f"[T2MVis] failed to save feats for {keyid}: {exc}")

            # npy-only mode: joints/features are already on disk, skip all rendering.
            if not self.render_video:
                continue

            gt_path = os.path.join(sample_dir, 'gt.mp4')
            pred_path = os.path.join(sample_dir, 'pred.mp4')

            if self._has_pair(sample_dir):
                self._skipped_existing += 1
                continue

            # First task: run synchronously in this process to surface any
            # worker-side import / ffmpeg errors immediately instead of
            # discovering them only at on_test_end.
            if not self._probed:
                self._probed = True
                try:
                    if not os.path.exists(gt_path):
                        render_fast_to_file(ref_np, gt_path, self.fps)
                    if not os.path.exists(pred_path):
                        render_fast_to_file(rst_np, pred_path, self.fps)
                    self._log(f"[T2MVis] Probe render OK for {keyid}.")
                except Exception as exc:
                    self._log(f"[T2MVis] Probe render FAILED for {keyid}: {exc!r}. "
                              f"Aborting visualization.")
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor = None
                    return
                continue

            if not os.path.exists(gt_path):
                self._futures.append(
                    self._executor.submit(render_fast_to_file, ref_np, gt_path, self.fps))
                self._submitted += 1
            if not os.path.exists(pred_path):
                self._futures.append(
                    self._executor.submit(render_fast_to_file, rst_np, pred_path, self.fps))
                self._submitted += 1

        # Advance global sample counter regardless of subsampling outcome.
        self._global_sample_idx += B

        # Periodically surface completed/failed futures so progress is visible
        # and worker errors are reported early.
        if (batch_idx + 1) % 10 == 0 and self._futures:
            still_pending = []
            for fut in self._futures:
                if fut.done():
                    try:
                        fut.result()
                        self._completed += 1
                    except Exception as exc:
                        self._log(f"[T2MVis] render job failed: {exc!r}")
                else:
                    still_pending.append(fut)
            self._futures = still_pending
            self._log(f"[T2MVis] progress: completed={self._completed} "
                      f"pending={len(self._futures)} submitted_total={self._submitted}")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        if self._executor is None:
            return
        total = len(self._futures)
        self._log(f"[T2MVis] Waiting for {total} render jobs to finish "
                  f"(submitted={self._submitted}, skipped_existing={self._skipped_existing}) ...")
        failures = 0
        from concurrent.futures import as_completed
        try:
            from tqdm import tqdm
            iterator = tqdm(as_completed(self._futures), total=total,
                            desc="[T2MVis] rendering", unit="clip")
        except Exception:
            iterator = as_completed(self._futures)
        for fut in iterator:
            try:
                fut.result()
            except Exception as exc:
                failures += 1
                self._log(f"[T2MVis] render job failed: {exc}")
        self._executor.shutdown(wait=True)
        self._executor = None
        self._futures = []
        self._log(f"[T2MVis] Done. failures={failures}. Output: {self.output_dir}")


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "PredLatRec": "train/pred_lat_rec",
        "DecoderOnly": "train/decoder_only_loss",
        "TokAcc": "Metrics/Token_Accuracy",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    # Save 10 latest checkpoints
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True,
        'save_on_train_epoch_end': True
    }
    callbacks.append(ModelCheckpoint(**checkpointParams))

    # Save checkpoint every n*5 epochs
    checkpointParams.update({
        'every_n_epochs':
        cfg.LOGGER.VAL_EVERY_STEPS*5,
        'save_top_k':
        -1,
        'save_last':
        False
    })
    callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'M2TMetrics': {
            'Metrics/Matching_score':{
                'abbr': 'MMDist',
                'mode': 'min'
            },
            'Metrics/R_precision_top_3': {
                'abbr': 'R3',
                'mode': 'max'
            }
        },
        'TM2TMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            },
            'Metrics/R_precision_top_3': {
                'abbr': 'R3',
                'mode': 'max'
            }
        },
        'MRMetrics': {
            'Metrics/MPJPE': {
                'abbr': 'MPJPE',
                'mode': 'min'
            }
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    checkpointParams.update({
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
    })

    # Lightning requires every ModelCheckpoint to have a unique state_key
    # (derived from monitor + mode + timing). Two metrics can request the same
    # monitored quantity -- e.g. both TM2TMetrics and M2TMetrics map to
    # R_precision_top_3 (max), as in the GNU phase -- which would otherwise crash
    # Trainer init. Dedupe on (monitor, mode), keeping the first occurrence.
    seen_monitors = set()
    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']
            elif 'MotionX' in cfg.DATASET.target and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                mode = metric_monitor_map[metric][metric_monitor]['mode']
                if (metric_monitor, mode) in seen_monitors:
                    continue
                seen_monitors.add((metric_monitor, mode))
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{ep}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    mode,
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def _get_metric_value(self, losses_dict, metric_key):
        if metric_key not in losses_dict:
            return None
        value = losses_dict[metric_key]
        return value.item() if hasattr(value, 'item') else value

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if isinstance(dico_name, (tuple, list)):
                    if len(dico_name) != 2:
                        continue
                    pred_value = self._get_metric_value(losses_dict, dico_name[0])
                    gt_value = self._get_metric_value(losses_dict, dico_name[1])
                    if pred_value is None or gt_value is None:
                        continue
                    metric = (
                        f"{metric_name} "
                        f"{metric_format.format(pred_value)}/{metric_format.format(gt_value)}"
                    )
                    metrics_str.append(metric)
                else:
                    metric = self._get_metric_value(losses_dict, dico_name)
                    if metric is None:
                        continue
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
