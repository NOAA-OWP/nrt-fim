import torch
import torch.nn as nn
from typing import Sequence, Optional

class MSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_values: Optional[Sequence[float]] = None,
        large_threshold: Optional[float] = None,
    ):
        """
        Mean-squared error using a mask derived ONLY from `target`.

        Masked (ignored) pixels are those where:
          - target is NaN or Inf
          - target equals any value in `ignore_values` (e.g., 255, 1e20)
          - (optional) abs(target) > large_threshold

        pred non-finite values at VALID locations are replaced with 0 before loss
        to avoid propagating NaN/Inf (mask is still target-only).

        Args:
          reduction: "mean" or "sum"
          ignore_values: list of exact values to ignore in target (e.g., [255, 1e20])
          large_threshold: ignore target pixels where |target| > large_threshold
        """
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction
        self.ignore_values = list(ignore_values) if ignore_values is not None else []
        self.large_threshold = large_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_stats: bool = False,
    ):
        """
        forwarding pred and target to calculate loss value

        :param pred: prediction
        :param target: satellite observation
        :param return_stats: if true, it returns
        :return:
        """
        pred = pred.float()
        target = target.float()

        # Build BAD mask from TARGET ONLY
        bad = ~torch.isfinite(target)  # NaN or Inf in target

        for v in self.ignore_values:
            # exact match masking (float compares are fine for sentinels like 255, 1e20)
            bad = bad | (target == float(v))

        if self.large_threshold is not None:
            bad = bad | (target.abs() > float(self.large_threshold))

        valid = ~bad

        # Edge case: no valid pixels
        if not torch.any(valid):
            loss = pred.new_tensor(0.0)
            if return_stats:
                stats = {
                    "num_valid": 0,
                    "num_bad": int(bad.sum().item()),
                    "tgt_nonfinite": int((~torch.isfinite(target)).sum().item()),
                    "tgt_equals_ignored": int(sum((target == float(v)).sum().item() for v in self.ignore_values)),
                    "tgt_gt_thresh": int((target.abs() > float(self.large_threshold)).sum().item()) if self.large_threshold is not None else 0,
                    "pred_nonfinite_in_valid": 0,
                }
                return loss, stats
            return loss

        # Sanitize pred at valid locations (keep mask target-only)
        pred_sanit = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

        # Compute masked MSE
        diff2 = (pred_sanit[valid] - target[valid]) ** 2
        loss = diff2.mean() if self.reduction == "mean" else diff2.sum()

        if return_stats:
            stats = {
                "num_valid": int(valid.sum().item()),
                "num_bad": int(bad.sum().item()),
                "tgt_nonfinite": int((~torch.isfinite(target)).sum().item()),
                "tgt_equals_ignored": int(sum((target == float(v)).sum().item() for v in self.ignore_values)),
                "tgt_gt_thresh": int((target.abs() > float(self.large_threshold)).sum().item()) if self.large_threshold is not None else 0,
                "pred_nonfinite_in_valid": int((~torch.isfinite(pred) & valid).sum().item()),
            }
            return loss, stats

        return loss


