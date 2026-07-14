# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch


def radius_point_nms(points, radius=8.0, max_det=300):
    """Non-maximum suppression for point predictions using distance threshold."""
    if points.shape[0] == 0:
        return points

    # Sort by confidence descending
    scores = points[:, 2]
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        # Compute distances between current point and remaining points
        xy_i = points[i, :2]
        xy_others = points[order[1:], :2]
        dists = torch.norm(xy_others - xy_i, dim=1)
        # Keep points beyond radius
        order = order[1:][dists > radius]

    return points[torch.tensor(keep, dtype=torch.long, device=points.device)]
