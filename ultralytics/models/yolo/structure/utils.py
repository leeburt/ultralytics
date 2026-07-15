# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.utils.ops import decode_port_distance, encode_port_distance


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


def associate_ports_to_components(
    components: torch.Tensor,
    ports: torch.Tensor,
    min_tolerance: float = 8.0,
    max_tolerance: float = 16.0,
    distance_ratio: float = 0.25,
    duplicate_cosine: float = 0.95,
    duplicate_radial_gap: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one linked port per component/ray and return links indexed into the retained ports.

    ``ports`` stores ``xy, confidence, predicted_component_xy, direction, rho``.  Candidates whose predicted
    endpoints do not reach a detected component are removed from final structure output; this prevents raw heatmap
    side-peaks from appearing as independent terminals.
    """
    link_columns = 7
    if components.numel() == 0 or ports.numel() == 0:
        return ports.new_zeros((0, ports.shape[-1])), ports.new_zeros((0, link_columns))

    endpoint_xy = ports[:, 3:5].float()
    component_xy = components[:, :2].float()
    endpoint_distances = torch.cdist(endpoint_xy, component_xy)
    endpoint_error, component_indices = endpoint_distances.min(dim=1)
    predicted_span = torch.norm(endpoint_xy - ports[:, :2].float(), dim=1)
    tolerances = (predicted_span * distance_ratio).clamp(min=min_tolerance, max=max_tolerance)
    candidate_indices = torch.nonzero(endpoint_error <= tolerances, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return ports.new_zeros((0, ports.shape[-1])), ports.new_zeros((0, link_columns))

    # Resolve same-terminal heatmap side-peaks after their parent component is known.
    kept_indices = []
    kept_rays = []
    order = candidate_indices[ports[candidate_indices, 2].argsort(descending=True)]
    for port_idx in order.tolist():
        component_idx = int(component_indices[port_idx])
        ray = ports[port_idx, :2].float() - component_xy[component_idx]
        radial_distance = ray.norm()
        ray = ray / radial_distance.clamp_min(1e-8)
        duplicate = any(
            previous_component == component_idx
            and torch.dot(ray, previous_ray) >= duplicate_cosine
            and torch.abs(radial_distance - previous_radial_distance) >= duplicate_radial_gap
            for previous_component, previous_ray, previous_radial_distance in kept_rays
        )
        if not duplicate:
            kept_indices.append(port_idx)
            kept_rays.append((component_idx, ray, radial_distance))

    kept_indices = torch.tensor(sorted(kept_indices), device=ports.device, dtype=torch.long)
    kept_ports = ports[kept_indices]
    original_to_kept = {int(original): new for new, original in enumerate(kept_indices.tolist())}
    links = []
    for original_idx in kept_indices.tolist():
        component_idx = component_indices[original_idx]
        tolerance = tolerances[original_idx]
        score = torch.sqrt(ports[original_idx, 2] * components[component_idx, 2])
        score = score * torch.exp(-0.5 * (endpoint_error[original_idx] / tolerance) ** 2)
        links.append(torch.stack((
            ports.new_tensor(float(original_to_kept[original_idx])),
            component_idx.to(dtype=ports.dtype),
            score.to(dtype=ports.dtype),
            ports[original_idx, 5],
            ports[original_idx, 6],
            ports[original_idx, 3],
            ports[original_idx, 4],
        )))
    return kept_ports, torch.stack(links)


def scale_link_geometry(links: torch.Tensor, scaled_ports: torch.Tensor) -> torch.Tensor:
    """Rebuild link endpoint and direction fields from ports already scaled to original image coordinates."""
    scaled_links = links.clone()
    if scaled_links.numel() == 0 or scaled_ports.numel() == 0:
        return scaled_links

    port_indices = scaled_links[:, 0].long()
    valid = (port_indices >= 0) & (port_indices < scaled_ports.shape[0])
    if valid.any():
        indices = port_indices[valid]
        port_xy = scaled_ports[indices, :2]
        endpoint_xy = scaled_ports[indices, 3:5]
        direction = endpoint_xy - port_xy
        scaled_links[valid, 3:5] = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scaled_links[valid, 5:7] = endpoint_xy
    return scaled_links
