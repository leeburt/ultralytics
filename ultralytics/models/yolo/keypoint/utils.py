# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch


def radius_point_nms(
    pred: torch.Tensor,
    radius: float,
    max_det: int,
    pre_nms_topk: int | None = None,
) -> torch.Tensor:
    """Keep the highest confidence point within each local radius."""
    if pred.numel() == 0:
        return pred

    radius2 = float(radius) ** 2
    max_det = max(int(max_det), 0)
    if max_det == 0:
        return pred[:0]

    pre_nms_topk = max_det * 10 if pre_nms_topk is None else int(pre_nms_topk)
    keep = []
    group_col = 3 if pred.shape[1] > 3 else None
    groups = pred[:, group_col].unique(sorted=True) if group_col is not None else pred.new_tensor([0])

    for group in groups:
        group_mask = pred[:, group_col] == group if group_col is not None else torch.ones(
            pred.shape[0], device=pred.device, dtype=torch.bool
        )
        group_idx = group_mask.nonzero(as_tuple=False).view(-1)
        group_pred = pred[group_idx]
        order = group_pred[:, 2].argsort(descending=True)
        if pre_nms_topk > 0:
            order = order[:pre_nms_topk]
        group_idx = group_idx[order]

        while group_idx.numel() and len(keep) < max_det:
            current = group_idx[0]
            keep.append(current)
            if group_idx.numel() == 1:
                break
            rest = group_idx[1:]
            dist2 = ((pred[rest, :2] - pred[current, :2]) ** 2).sum(1)
            group_idx = rest[dist2 > radius2]

        if len(keep) >= max_det:
            break

    if not keep:
        return pred[:0]

    keep_idx = torch.stack(keep)
    keep_idx = keep_idx[pred[keep_idx, 2].argsort(descending=True)[:max_det]]
    return pred[keep_idx]
