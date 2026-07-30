"""
Token Consistency Metrics for evaluating motion token generation quality.

Computes token-level metrics between generated and ground truth motion token sequences:
- Token_Accuracy: positional exact match rate (truncated to shorter sequence)
- Token_Levenshtein: normalized edit distance (0=identical, 1=fully different)
- Token_LengthRatio: mean(len(gen) / len(gt)), ideal=1.0
- Token_F1: token-level F1 score (set-based precision & recall)
"""

from collections import Counter

import torch
from torch import Tensor
from torchmetrics import Metric
from typing import List


def _levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute Levenshtein (edit) distance between two integer sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use two-row DP to save memory
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


class TokenConsistencyMetrics(Metric):
    """
    Token-level consistency metrics for motion generation.
    
    Accumulates (gt_tokens, gen_tokens) pairs across batches,
    then computes aggregate metrics at epoch end via compute().
    """

    full_state_update = False

    def __init__(self, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.metrics = [
            "Token_Accuracy",
            "Token_Levenshtein",
            "Token_LengthRatio",
            "Token_F1",
        ]

        # Running sums for incremental computation
        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_compared", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("sum_edit_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_length_ratio", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, gt_tokens_list: List[List[int]], gen_tokens_list: List[List[int]]):
        """
        Update metrics with a batch of (gt, gen) token sequence pairs.
        
        Args:
            gt_tokens_list: List of ground truth token sequences (each is List[int])
            gen_tokens_list: List of generated token sequences (each is List[int])
        """
        assert len(gt_tokens_list) == len(gen_tokens_list), \
            f"Mismatched lengths: {len(gt_tokens_list)} vs {len(gen_tokens_list)}"

        for gt_seq, gen_seq in zip(gt_tokens_list, gen_tokens_list):
            if len(gt_seq) == 0:
                continue

            self.count += 1

            # --- Position-independent accuracy (multiset matching) ---
            gen_counter = Counter(gen_seq)
            correct = 0
            for tok in gt_seq:
                if gen_counter[tok] > 0:
                    correct += 1
                    gen_counter[tok] -= 1
            self.total_correct += correct
            self.total_compared += len(gt_seq)  # normalize by GT length

            # --- Normalized Levenshtein distance ---
            max_len = max(len(gt_seq), len(gen_seq))
            edit_dist = _levenshtein_distance(gt_seq, gen_seq)
            self.sum_edit_dist += edit_dist / max_len  # normalized to [0, 1]

            # --- Length ratio ---
            self.sum_length_ratio += len(gen_seq) / len(gt_seq)

            # --- Token F1 (set-based) ---
            gt_set = set(gt_seq)
            gen_set = set(gen_seq)
            if len(gen_set) > 0 and len(gt_set) > 0:
                tp = len(gt_set & gen_set)
                precision = tp / len(gen_set)
                recall = tp / len(gt_set)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
            self.sum_f1 += f1

    def compute(self, sanity_flag=False):
        count = self.count.item()

        if sanity_flag or count == 0:
            return {m: torch.tensor(0.0) for m in self.metrics}

        total_compared = self.total_compared.item()
        accuracy = self.total_correct.float() / max(total_compared, 1)
        levenshtein = self.sum_edit_dist / count
        length_ratio = self.sum_length_ratio / count
        f1 = self.sum_f1 / count

        return {
            "Token_Accuracy": accuracy,
            "Token_Levenshtein": levenshtein,
            "Token_LengthRatio": length_ratio,
            "Token_F1": f1,
        }
