# Structure CenterNet Polar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement a new "structure" task in Ultralytics for predicting circuit components, ports, and their polar-form relationships.

**Architecture:** Follows the keypoint task pattern with:
- StructureHeatmap head (component/port heatmaps + polar relationship)
- StructureHeatmapLoss (focal + offset + polar relation losses)
- StructureTrainer/Validator/Predictor
- Data converter for device_ports dataset

**Tech Stack:** PyTorch, Ultralytics YOLO11 architecture

## Global Constraints

- Base branch: exp/keypoint-centernet-strict-loss
- Single-class only (component and port are single categories)
- Kpt_shape = [25, 3] (1 component center + 24 generic ports)
- Output stride = 4
- Polar form: direction (cos/sin) + distance (log(1+d))
- Loss weights: 0.5 * dir_loss + 0.5 * rho_loss + 0.25 * endpoint_loss
- Relation loss ramp-up: 0 to full weight over first 5 epochs

---

## Task 1: Create StructureHeatmap Head Module

**Files:**
- Modify: `/data-ssd/libo/ultralytics/ultralytics/nn/modules/head.py`

**Interfaces:**
- Consumes: Multi-scale features (P2-P5) from backbone
- Produces: `component_hm`, `component_off`, `port_hm`, `port_off`, `direction`, `rho`

### Step 1.1: Write the StructureHeatmap class
Add after KeypointHeatmap class (line ~408):
```python
class StructureHeatmap(nn.Module):
    """CenterNet-style structure prediction head with component/port heatmaps and polar port->component relationships."""

    export = False
    format = None
    max_det = 1024

    def __init__(
        self,
        kpt_shape=(25, 3),  # 1 component center + 24 ports
        topk_component=128,
        topk_port=384,
        ch=(),
    ):
        """Initialize lateral fusion plus heatmap, offset, and relation heads."""
        super().__init__()
        self.kpt_shape = kpt_shape
        self.topk_component = topk_component
        self.topk_port = topk_port
        self.nl = 1
        self.stride = torch.zeros(self.nl)
        ch = list(ch) if isinstance(ch, (list, tuple)) else [ch]
        c = max(ch[0] // 4, 64, 32)
        self.lateral = nn.ModuleList(Conv(x, c, 1) for x in ch)
        self.refine = nn.Sequential(Conv(c, c, 3), Conv(c, c, 3))

        # Component head
        self.component_hm = nn.Conv2d(c, 1, 1)
        self.component_off = nn.Conv2d(c, 2, 1)

        # Port head
        self.port_hm = nn.Conv2d(c, 1, 1)
        self.port_off = nn.Conv2d(c, 2, 1)

        # Relation head (polar form: direction + rho)
        self.direction = nn.Conv2d(c, 2, 1)  # cos/sin
        self.rho = nn.Conv2d(c, 1, 1)  # log(1+distance)

    def forward(self, x):
        """Return raw predictions during training, decoded candidates during inference."""
        feats = list(x) if isinstance(x, (list, tuple)) else [x]
        base = self.lateral[0](feats[0])
        for i, feat in enumerate(feats[1:], 1):
            base = base + F.interpolate(self.lateral[i](feat), size=base.shape[2:], mode="nearest")
        feat = self.refine(base)
        preds = {
            "component_hm": self.component_hm(feat),
            "component_off": self.component_off(feat),
            "port_hm": self.port_hm(feat),
            "port_off": self.port_off(feat),
            "direction": self.direction(feat),
            "rho": self.rho(feat),
            "feats": feats,
        }
        if self.training:
            return preds
        decoded = self.decode(preds)
        return decoded if self.export else (decoded, preds)

    def decode(self, preds):
        """Decode heatmap maxima, offsets, and relations to image-space candidates."""
        device = preds["component_hm"].device
        dtype = preds["component_hm"].dtype
        stride = self.stride.to(device=device, dtype=dtype).view(-1)[0]

        # Decode components
        component_hm = preds["component_hm"].sigmoid()
        component_off = preds["component_off"]
        component_pooled = F.max_pool2d(component_hm, kernel_size=3, stride=1, padding=1)
        component_peaks = component_hm * (component_hm == component_pooled)
        bs, _, h, w = component_peaks.shape
        total = h * w
        k_comp = min(int(self.max_det), int(self.topk_component), total)
        comp_scores, comp_inds = component_peaks.view(bs, -1).topk(k_comp, dim=1)
        comp_ys = comp_inds // w
        comp_xs = comp_inds % w
        comp_xy = torch.stack((comp_xs.to(dtype), comp_ys.to(dtype)), -1)
        comp_off_gather = component_off.permute(0, 2, 3, 1).reshape(bs, -1, 2)
        comp_off = comp_off_gather.gather(1, comp_inds.unsqueeze(-1).expand(-1, -1, 2))
        comp_xy = (comp_xy + comp_off) * stride

        # Decode ports
        port_hm = preds["port_hm"].sigmoid()
        port_off = preds["port_off"]
        direction = preds["direction"]
        rho = preds["rho"]
        port_pooled = F.max_pool2d(port_hm, kernel_size=3, stride=1, padding=1)
        port_peaks = port_hm * (port_hm == port_pooled)
        k_port = min(int(self.max_det), int(self.topk_port), total)
        port_scores, port_inds = port_peaks.view(bs, -1).topk(k_port, dim=1)
        port_ys = port_inds // w
        port_xs = port_inds % w
        port_xy = torch.stack((port_xs.to(dtype), port_ys.to(dtype)), -1)
        port_off_gather = port_off.permute(0, 2, 3, 1).reshape(bs, -1, 2)
        port_off = port_off_gather.gather(1, port_inds.unsqueeze(-1).expand(-1, -1, 2))
        port_xy = (port_xy + port_off) * stride

        # Gather relation predictions at port locations
        dir_gather = direction.permute(0, 2, 3, 1).reshape(bs, -1, 2)
        rho_gather = rho.permute(0, 2, 3, 1).reshape(bs, -1, 1)
        port_dir = dir_gather.gather(1, port_inds.unsqueeze(-1).expand(-1, -1, 2))
        port_rho = rho_gather.gather(1, port_inds.unsqueeze(-1))

        # Compute predicted component endpoints from ports
        port_dir_norm = port_dir / (port_dir.norm(dim=-1, keepdim=True) + 1e-8)
        distance = torch.expm1(F.softplus(port_rho))  # exp(rho) - 1
        pred_component_xy = port_xy + distance * stride * port_dir_norm

        # Format output: [component_xy, comp_score, port_xy, port_score, pred_component_xy, dir, rho]
        # We return two separate tensors for easier post-processing
        return {
            "components": torch.cat((comp_xy, comp_scores.unsqueeze(-1)), -1),  # (bs, k_comp, 3)
            "ports": torch.cat((port_xy, port_scores.unsqueeze(-1), pred_component_xy, port_dir, port_rho), -1),  # (bs, k_port, 3+2+2+1=8)
        }

    def bias_init(self):
        """Initialize heatmap confidence low and offsets near zero."""
        self.component_hm.bias.data.fill_(math.log(0.01 / 0.99))
        self.port_hm.bias.data.fill_(math.log(0.01 / 0.99))
        self.component_off.bias.data.zero_()
        self.port_off.bias.data.zero_()
        self.direction.bias.data.zero_()
        self.rho.bias.data.zero_()
```

### Step 1.2: Verify imports
Check that `math` and `torch.nn.functional as F` are imported at top of file.

### Step 1.3: Commit
```bash
cd /data-ssd/libo/ultralytics
git checkout -b exp/structure-centernet-polar
git add ultralytics/nn/modules/head.py
git commit -m "feat: Add StructureHeatmap head module"
```

---

## Task 2: Create StructureHeatmapLoss Module

**Files:**
- Modify: `/data-ssd/libo/ultralytics/ultralytics/utils/loss.py`

**Interfaces:**
- Consumes: Model predictions dict + batch dict
- Produces: Scalar loss + detached loss items

### Step 2.1: Write the StructureHeatmapLoss class
Add after KeypointHeatmapLoss class (line ~533):
```python
class StructureHeatmapLoss:
    """CenterNet-style structure loss with component/port heatmaps and polar port->component relations."""

    def __init__(self, model):
        """Initialize structure loss state from the model head."""
        self.head = model.model[-1]
        self.component_hm_weight = 1.0
        self.component_off_weight = 1.0
        self.port_hm_weight = 1.0
        self.port_off_weight = 1.0
        self.dir_weight = 0.5
        self.rho_weight = 0.5
        self.endpoint_weight = 0.25
        self.ramp_epochs = 5
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Set current epoch for loss weight ramp-up."""
        self.current_epoch = epoch

    def __call__(self, preds, batch):
        """Compute structure losses."""
        if isinstance(preds, (tuple, list)):
            preds = preds[1]
        device = preds["component_hm"].device
        dtype = preds["component_hm"].dtype
        bs, _, h, w = preds["component_hm"].shape
        stride = self.head.stride.to(device=device, dtype=dtype).view(-1)[0]

        # Initialize targets
        target_component_hm = torch.zeros_like(preds["component_hm"])
        target_component_off = torch.zeros((bs, 2, h, w), device=device, dtype=dtype)
        component_off_mask = torch.zeros((bs, h, w), device=device, dtype=torch.bool)

        target_port_hm = torch.zeros_like(preds["port_hm"])
        target_port_off = torch.zeros((bs, 2, h, w), device=device, dtype=dtype)
        port_off_mask = torch.zeros((bs, h, w), device=device, dtype=torch.bool)

        target_dir = torch.zeros((bs, 2, h, w), device=device, dtype=dtype)
        target_rho = torch.zeros((bs, 1, h, w), device=device, dtype=dtype)
        relation_mask = torch.zeros((bs, h, w), device=device, dtype=torch.bool)

        # Collision counters for stats
        component_collisions = 0
        port_collisions = 0

        # Build targets from keypoint annotations
        keypoints = batch.get("keypoints")
        batch_idx = batch.get("batch_idx")
        bboxes = batch.get("bboxes")
        if keypoints is not None and batch_idx is not None and keypoints.numel():
            keypoints = keypoints.to(device=device, dtype=dtype)
            batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)
            bboxes = bboxes.to(device=device, dtype=dtype) if bboxes is not None and bboxes.numel() else None
            imgsz = torch.tensor((h, w), device=device, dtype=dtype) * stride

            for i in range(keypoints.shape[0]):
                b = int(batch_idx[i].item())
                if b < 0 or b >= bs:
                    continue

                # Keypoint 0 is component center, keypoints 1-24 are ports
                kpts = keypoints[i]
                component_kpt = kpts[0]
                port_kpts = kpts[1:]

                # Draw component center
                if component_kpt[2] > 0:
                    comp_xy_abs = component_kpt[:2].clone()
                    comp_xy_abs[0] *= imgsz[1]
                    comp_xy_abs[1] *= imgsz[0]
                    comp_xy = comp_xy_abs / stride
                    cx = comp_xy[0].clamp(0, max(float(w) - 1e-4, 0.0))
                    cy = comp_xy[1].clamp(0, max(float(h) - 1e-4, 0.0))
                    cxi, cyi = int(cx.floor().item()), int(cy.floor().item())

                    # Compute radius from bbox if available
                    radius = 2  # fallback
                    if bboxes is not None and i < bboxes.shape[0]:
                        box_w = float((bboxes[i, 2] * imgsz[1] / stride).clamp(min=0).item())
                        box_h = float((bboxes[i, 3] * imgsz[0] / stride).clamp(min=0).item())
                        radius = max(1, min(8, int(self._gaussian_radius((box_h, box_w)))))

                    self._draw_gaussian(target_component_hm[b, 0], cxi, cyi, radius)
                    if component_off_mask[b, cyi, cxi]:
                        component_collisions += 1
                    else:
                        target_component_off[b, 0, cyi, cxi] = cx - cxi
                        target_component_off[b, 1, cyi, cxi] = cy - cyi
                        component_off_mask[b, cyi, cxi] = True

                # Draw ports and their relation to component center
                for port_kpt in port_kpts:
                    if port_kpt[2] > 0:
                        port_xy_abs = port_kpt[:2].clone()
                        port_xy_abs[0] *= imgsz[1]
                        port_xy_abs[1] *= imgsz[0]
                        port_xy = port_xy_abs / stride
                        px = port_xy[0].clamp(0, max(float(w) - 1e-4, 0.0))
                        py = port_xy[1].clamp(0, max(float(h) - 1e-4, 0.0))
                        pxi, pyi = int(px.floor().item()), int(py.floor().item())

                        # Port heatmap uses fixed small radius
                        self._draw_gaussian(target_port_hm[b, 0], pxi, pyi, 1)
                        if port_off_mask[b, pyi, pxi]:
                            port_collisions += 1
                        else:
                            target_port_off[b, 0, pyi, pxi] = px - pxi
                            target_port_off[b, 1, pyi, pxi] = py - pyi
                            port_off_mask[b, pyi, pxi] = True

                        # Relation target (only if component is visible)
                        if component_kpt[2] > 0:
                            delta_xy = comp_xy - port_xy  # in feature cells
                            distance_gt = delta_xy.norm()
                            if distance_gt >= 0.5:  # stable direction threshold
                                direction_gt = delta_xy / distance_gt
                                rho_gt = torch.log1p(distance_gt)  # log(1+d)
                                if not relation_mask[b, pyi, pxi]:
                                    target_dir[b, :, pyi, pxi] = direction_gt
                                    target_rho[b, 0, pyi, pxi] = rho_gt
                                    relation_mask[b, pyi, pxi] = True

        # Compute losses
        component_hm_loss = self._focal_loss(preds["component_hm"], target_component_hm)
        port_hm_loss = self._focal_loss(preds["port_hm"], target_port_hm)

        if component_off_mask.any():
            component_off_loss = F.l1_loss(
                preds["component_off"].permute(0, 2, 3, 1)[component_off_mask],
                target_component_off.permute(0, 2, 3, 1)[component_off_mask],
                reduction="sum",
            )
            component_off_loss = component_off_loss / (component_off_mask.sum().clamp(min=1).to(dtype) * 2.0)
        else:
            component_off_loss = preds["component_off"].sum() * 0.0

        if port_off_mask.any():
            port_off_loss = F.l1_loss(
                preds["port_off"].permute(0, 2, 3, 1)[port_off_mask],
                target_port_off.permute(0, 2, 3, 1)[port_off_mask],
                reduction="sum",
            )
            port_off_loss = port_off_loss / (port_off_mask.sum().clamp(min=1).to(dtype) * 2.0)
        else:
            port_off_loss = preds["port_off"].sum() * 0.0

        # Polar relation losses
        dir_loss = torch.tensor(0.0, device=device, dtype=dtype)
        rho_loss = torch.tensor(0.0, device=device, dtype=dtype)
        endpoint_loss = torch.tensor(0.0, device=device, dtype=dtype)

        if relation_mask.any():
            # Direction loss (cosine similarity)
            pred_dir = preds["direction"].permute(0, 2, 3, 1)[relation_mask]
            gt_dir = target_dir.permute(0, 2, 3, 1)[relation_mask]
            pred_dir_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + 1e-8)
            dir_loss = 1.0 - (pred_dir_norm * gt_dir).sum(dim=-1).mean()

            # Rho loss (smooth L1)
            pred_rho = preds["rho"].permute(0, 2, 3, 1)[relation_mask]
            gt_rho = target_rho.permute(0, 2, 3, 1)[relation_mask]
            rho_loss = F.smooth_l1_loss(pred_rho, gt_rho)

            # Endpoint loss (reconstructed component center)
            pred_distance = torch.expm1(F.softplus(pred_rho))  # exp(rho) - 1
            pred_delta = pred_distance * pred_dir_norm
            # gt_delta in feature cells
            gt_distance = torch.expm1(gt_rho)
            gt_delta = gt_distance * gt_dir
            # Normalize by gt distance for scale-invariance
            norm = gt_distance.clamp(min=1.0)
            endpoint_loss = F.smooth_l1_loss(pred_delta / norm, gt_delta / norm)

        # Ramp up relation loss weights
        ramp_factor = min(1.0, self.current_epoch / max(1, self.ramp_epochs))
        dir_weight = self.dir_weight * ramp_factor
        rho_weight = self.rho_weight * ramp_factor
        endpoint_weight = self.endpoint_weight * ramp_factor

        # Total loss
        loss = (
            component_hm_loss * self.component_hm_weight
            + component_off_loss * self.component_off_weight
            + port_hm_loss * self.port_hm_weight
            + port_off_loss * self.port_off_weight
            + dir_loss * dir_weight
            + rho_loss * rho_weight
            + endpoint_loss * endpoint_weight
        )

        loss_items = torch.stack((
            component_hm_loss * self.component_hm_weight,
            component_off_loss * self.component_off_weight,
            port_hm_loss * self.port_hm_weight,
            port_off_loss * self.port_off_weight,
            dir_loss * dir_weight,
            rho_loss * rho_weight,
            endpoint_loss * endpoint_weight,
        ))
        return loss * bs, loss_items.detach()

    @staticmethod
    def _focal_loss(pred, gt):
        """Modified focal loss used by CenterNet."""
        pred = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = gt.eq(1)
        neg = gt.lt(1)
        neg_weights = (1 - gt).pow(4)
        pos_loss = -(pred.log() * (1 - pred).pow(2) * pos).sum()
        neg_loss = -((1 - pred).log() * pred.pow(2) * neg_weights * neg).sum()
        num_pos = pos.sum().clamp(min=1)
        return (pos_loss + neg_loss) / num_pos

    @staticmethod
    def _draw_gaussian(heatmap, x, y, radius):
        """Draw a small Gaussian peak on a heatmap in-place."""
        diameter = 2 * radius + 1
        xs = torch.arange(diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * (diameter / 6) ** 2))

        height, width = heatmap.shape
        left, right = min(x, radius), min(width - x - 1, radius)
        top, bottom = min(y, radius), min(height - y - 1, radius)
        patch = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
        gpatch = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
        torch.maximum(patch, gpatch, out=patch)

    @staticmethod
    def _gaussian_radius(det_size, min_overlap=0.7):
        """Compute CenterNet Gaussian radius from object size on the output feature map."""
        height, width = det_size
        if height <= 0 or width <= 0:
            return 0.0

        a1 = 1.0
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(max(torch.tensor(0.0), b1**2 - 4 * a1 * c1))
        r1 = (b1 + sq1) / 2

        a2 = 4.0
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(max(torch.tensor(0.0), b2**2 - 4 * a2 * c2))
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(max(torch.tensor(0.0), b3**2 - 4 * a3 * c3))
        r3 = (b3 + sq3) / 2
        return float(min(r1, r2, r3))
```

### Step 2.2: Commit
```bash
cd /data-ssd/libo/ultralytics
git add ultralytics/utils/loss.py
git commit -m "feat: Add StructureHeatmapLoss module"
```

---

## Task 3: Create StructureModel in tasks.py

**Files:**
- Modify: `/data-ssd/libo/ultralytics/ultralytics/nn/tasks.py`

**Interfaces:**
- Consumes: YAML cfg, channels, num_classes
- Produces: StructureModel with StructureHeatmap head

### Step 3.1: Add imports
At top of file, add:
```python
from ultralytics.nn.modules import StructureHeatmap
```
(Should be near KeypointHeatmap import if it exists)

### Step 3.2: Add StructureModel class
Add after KeypointModel class (line ~748):
```python
class StructureModel(BaseModel):
    """YOLO structure model that predicts components, ports, and their polar relationships."""

    def __init__(self, cfg="yolo11n-structure.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize a structure model."""
        super().__init__()
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        m = self.model[-1]
        if isinstance(m, StructureHeatmap):
            self.kpt_shape = m.kpt_shape
            s = 256
            m.inplace = self.inplace
            self.model.eval()
            m.training = True
            output = self.forward(torch.zeros(1, ch, s, s))
            m.stride = torch.tensor([s / output["component_hm"].shape[-2]])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Initialize the loss criterion for the StructureModel."""
        return StructureHeatmapLoss(self)
```

### Step 3.3: Add StructureHeatmap import to parse_model
Find where `KeypointHeatmap` is handled in `parse_model` function and add similar handling for `StructureHeatmap`.

### Step 3.4: Commit
```bash
cd /data-ssd/libo/ultralytics
git add ultralytics/nn/tasks.py
git commit -m "feat: Add StructureModel class"
```

---

## Task 4: Create Structure Task Directory and Files

**Files:**
- Create: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/structure/__init__.py`
- Create: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/structure/train.py`
- Create: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/structure/val.py`
- Create: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/structure/predict.py`
- Create: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/structure/utils.py`

### Step 4.1: Create utils.py
```python
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
```

### Step 4.2: Create predict.py
```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from .utils import radius_point_nms


class StructurePredictor(BasePredictor):
    """Predictor for structure models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize structure predictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "structure"

    def postprocess(self, preds, img, orig_imgs):
        """Convert structure predictions to Results objects."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        component_nms_radius = 16.0
        port_nms_radius = 8.0
        max_det = self.args.max_det

        for i, (pred_dict, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]

            # Process components
            components = pred_dict["components"]
            comp_conf = components[:, 2]
            comp_keep = comp_conf >= self.args.conf
            components = components[comp_keep]
            if components.shape[0]:
                components = radius_point_nms(components, component_nms_radius, max_det)
                comp_xy = components[:, :3].view(-1, 1, 3)
                comp_xy = ops.scale_coords(img.shape[2:], comp_xy, orig_img.shape)
            else:
                comp_xy = torch.zeros((0, 1, 3), device=components.device)

            # Process ports
            ports = pred_dict["ports"]
            port_conf = ports[:, 2]
            port_keep = port_conf >= self.args.conf
            ports = ports[port_keep]
            if ports.shape[0]:
                ports = radius_point_nms(ports, port_nms_radius, max_det)
                port_xy = ports[:, :3].view(-1, 1, 3)
                port_xy = ops.scale_coords(img.shape[2:], port_xy, orig_img.shape)
                pred_comp_xy = ports[:, 3:5].unsqueeze(1)
                pred_comp_xy = ops.scale_coords(img.shape[2:], pred_comp_xy, orig_img.shape)
            else:
                port_xy = torch.zeros((0, 1, 3), device=ports.device)
                pred_comp_xy = torch.zeros((0, 1, 2), device=ports.device)

            # Store in results (using keypoints field for both components and ports)
            # We'll concatenate them with different visibility flags to distinguish
            results.append(Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                keypoints={
                    "components": comp_xy,
                    "ports": port_xy,
                    "pred_components": pred_comp_xy,
                }
            ))
        return results
```

### Step 4.3: Create train.py
```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.modules import StructureHeatmap
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import StructureModel
from ultralytics.utils import DEFAULT_CFG, RANK


def _unwrap(model):
    """Return the underlying nn.Module, unwrapping DDP if needed."""
    return model.module if hasattr(model, "module") else model


class StructureTrainer(DetectionTrainer):
    """Trainer for structure models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize structure trainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "structure"
        super().__init__(cfg, overrides, _callbacks)
        self.epoch = 0

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> StructureModel:
        """Get structure model with optional pretrained weights."""
        model = StructureModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach dataset metadata to the model."""
        super().set_model_attributes()
        unwrapped = _unwrap(self.model)
        unwrapped.kpt_shape = self.data["kpt_shape"]
        unwrapped.kpt_names = self.data.get("kpt_names") or {
            i: [str(j) for j in range(unwrapped.kpt_shape[0])] for i in range(unwrapped.nc)
        }

    def get_validator(self):
        """Return validator for structure models."""
        unwrapped = _unwrap(self.model)
        self.loss_names = (
            "comp_hm_loss",
            "comp_off_loss",
            "port_hm_loss",
            "port_off_loss",
            "dir_loss",
            "rho_loss",
            "endpoint_loss",
        ) if isinstance(unwrapped.model[-1], StructureHeatmap) else ("loss",)
        return yolo.structure.StructureValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Load dataset metadata and require keypoint shape."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            data["kpt_shape"] = [25, 3]  # Default: 1 component + 24 ports
        return data

    def _do_train(self, world_size=1):
        """Override to pass epoch to loss criterion."""
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            # Update loss epoch for ramp-up
            unwrapped = _unwrap(self.model)
            if hasattr(unwrapped, "criterion") and hasattr(unwrapped.criterion, "set_epoch"):
                unwrapped.criterion.set_epoch(epoch)
            # Continue with normal training
            super()._do_train(world_size)
```

### Step 4.4: Create val.py
```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.plotting import plot_images
from .utils import radius_point_nms


class StructureMetrics:
    """Metrics for structure prediction with component, port, and link evaluation."""

    fixed_thresholds = (0.25, 0.30, 0.40, 0.45)

    keys = [
        "metrics/component_8px_F1",
        "metrics/port_8px_F1",
        "metrics/link_8px_F1",
        "metrics/link_association_accuracy",
        "metrics/angle_MAE",
        "metrics/endpoint_MAE",
        "metrics/strict_object_recall",
        "fitness",
    ]

    def __init__(self):
        """Initialize counters."""
        self.reset()
        self.speed = None
        self.save_dir = None

    def reset(self):
        """Reset accumulated metrics."""
        self.records = []
        self.num_gt_components = 0
        self.num_gt_ports = 0
        self.num_gt_links = 0

    def update(self, pred_components, pred_ports, pred_links, gt_components, gt_ports, gt_links):
        """Store one image of predictions and ground-truth."""
        self.records.append((
            pred_components.detach().float().cpu(),
            pred_ports.detach().float().cpu(),
            pred_links.detach().float().cpu() if pred_links is not None else None,
            gt_components.detach().float().cpu(),
            gt_ports.detach().float().cpu(),
            gt_links.detach().float().cpu() if gt_links is not None else None,
        ))
        self.num_gt_components += int(gt_components.shape[0])
        self.num_gt_ports += int(gt_ports.shape[0])
        self.num_gt_links += int(gt_links.shape[0]) if gt_links is not None else 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return gather-able stats dict for DDP."""
        return {
            "records": self.records,
            "num_gt_components": self.num_gt_components,
            "num_gt_ports": self.num_gt_ports,
            "num_gt_links": self.num_gt_links,
        }

    @stats.setter
    def stats(self, value: dict[str, Any]) -> None:
        """Restore stats from gathered dict."""
        self.records = value["records"]
        self.num_gt_components = value["num_gt_components"]
        self.num_gt_ports = value["num_gt_ports"]
        self.num_gt_links = value["num_gt_links"]

    def clear_stats(self) -> None:
        """Clear accumulated stats after DDP gathering."""
        self.records = []
        self.num_gt_components = 0
        self.num_gt_ports = 0
        self.num_gt_links = 0

    def mean_results(self):
        """Return metrics in display order."""
        metrics = self._compute_metrics()
        # Fitness = 0.3 * component_F1 + 0.7 * link_F1
        fitness = 0.3 * metrics["component_8px_f1"] + 0.7 * metrics["link_8px_f1"]
        return [
            metrics["component_8px_f1"],
            metrics["port_8px_f1"],
            metrics["link_8px_f1"],
            metrics["link_association_accuracy"],
            metrics["angle_mae"],
            metrics["endpoint_mae"],
            metrics["strict_object_recall"],
            fitness,
        ]

    @property
    def results_dict(self):
        """Return metrics dict."""
        return dict(zip(self.keys, self.mean_results()))

    @staticmethod
    def _match_points(pred, gt, threshold):
        """Match predicted points to GT points with distance threshold."""
        if pred.numel() == 0:
            return [], list(range(gt.shape[0]))
        if gt.numel() == 0:
            return list(range(pred.shape[0])), []

        # Compute distance matrix
        pred_xy = pred[:, :2]
        gt_xy = gt[:, :2]
        dists = torch.cdist(pred_xy, gt_xy)

        # Greedily match highest confidence predictions first
        order = pred[:, 2].argsort(descending=True)
        matched_gt = set()
        matched_pred = set()
        matches = []

        for pred_idx in order:
            best_dist = float("inf")
            best_gt_idx = -1
            for gt_idx in range(gt.shape[0]):
                if gt_idx in matched_gt:
                    continue
                if dists[pred_idx, gt_idx] < threshold and dists[pred_idx, gt_idx] < best_dist:
                    best_dist = dists[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                matches.append((pred_idx, best_gt_idx))

        fp = [i for i in range(pred.shape[0]) if i not in matched_pred]
        fn = [i for i in range(gt.shape[0]) if i not in matched_gt]
        return matches, fp, fn

    def _compute_metrics(self):
        """Compute all structure metrics."""
        component_threshold = 8.0  # 2 feature cells at stride 4
        port_threshold = 8.0
        link_threshold = 8.0

        component_tp = 0
        component_fp = 0
        component_fn = 0
        port_tp = 0
        port_fp = 0
        port_fn = 0
        link_tp = 0
        link_fp = 0
        link_fn = 0
        correct_associations = 0
        angle_errors = []
        endpoint_errors = []
        strict_object_count = 0
        total_objects = 0

        for record in self.records:
            pred_comp, pred_port, pred_link, gt_comp, gt_port, gt_link = record

            # Component matching
            comp_matches, comp_fp_list, comp_fn_list = self._match_points(pred_comp, gt_comp, component_threshold)
            component_tp += len(comp_matches)
            component_fp += len(comp_fp_list)
            component_fn += len(comp_fn_list)

            # Port matching
            port_matches, port_fp_list, port_fn_list = self._match_points(pred_port, gt_port, port_threshold)
            port_tp += len(port_matches)
            port_fp += len(port_fp_list)
            port_fn += len(port_fn_list)

            # Build GT port->component links from data
            # For each GT port, find its GT component
            gt_port_to_comp = {}
            if gt_link is not None:
                for link in gt_link:
                    port_idx = int(link[0])
                    comp_idx = int(link[1])
                    gt_port_to_comp[port_idx] = comp_idx

            # For matched ports, check if their predicted component matches GT component
            matched_port_to_gt_port = {p_idx: g_idx for p_idx, g_idx in port_matches}
            matched_comp_to_gt_comp = {p_idx: g_idx for p_idx, g_idx in comp_matches}

            # For strict object recall: component with all ports matched and linked correctly
            gt_comp_port_counts = {}
            gt_comp_correct_port_counts = {}
            for link_idx in range(gt_link.shape[0]) if gt_link is not None else []:
                comp_idx = int(gt_link[link_idx, 1])
                gt_comp_port_counts[comp_idx] = gt_comp_port_counts.get(comp_idx, 0) + 1

            total_objects += len(gt_comp_port_counts)

            # Evaluate links from predictions
            if pred_link is not None:
                for link in pred_link:
                    pred_port_idx = int(link[0])
                    pred_comp_idx = int(link[1])
                    score = link[2]

                    # Check if port is matched to GT port
                    if pred_port_idx in matched_port_to_gt_port:
                        gt_port_idx = matched_port_to_gt_port[pred_port_idx]
                        # Check if this GT port should have a link
                        if gt_port_idx in gt_port_to_comp:
                            gt_comp_idx = gt_port_to_comp[gt_port_idx]
                            # Check if predicted component is matched to GT component
                            if pred_comp_idx in matched_comp_to_gt_comp:
                                matched_gt_comp_idx = matched_comp_to_gt_comp[pred_comp_idx]
                                if matched_gt_comp_idx == gt_comp_idx:
                                    link_tp += 1
                                    correct_associations += 1
                                    gt_comp_correct_port_counts[gt_comp_idx] = gt_comp_correct_port_counts.get(gt_comp_idx, 0) + 1
                                    # Compute angle error if available
                                    if link.shape[0] > 3:
                                        pred_angle = torch.atan2(link[4], link[3])
                                        gt_delta = gt_comp[gt_comp_idx, :2] - gt_port[gt_port_idx, :2]
                                        gt_angle = torch.atan2(gt_delta[1], gt_delta[0])
                                        angle_diff = torch.abs(pred_angle - gt_angle)
                                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                                        angle_errors.append(float(angle_diff))
                                    # Compute endpoint error if available
                                    if link.shape[0] > 5:
                                        pred_endpoint = link[5:7]
                                        gt_endpoint = gt_comp[gt_comp_idx, :2]
                                        endpoint_err = torch.norm(pred_endpoint - gt_endpoint)
                                        endpoint_errors.append(float(endpoint_err))
                                else:
                                    link_fp += 1
                            else:
                                link_fp += 1
                        else:
                            link_fp += 1
                    else:
                        link_fp += 1

            # Count FN links
            link_fn += self.num_gt_links - link_tp if self.num_gt_links > link_tp else 0

            # Check strict object recall
            for comp_idx in gt_comp_port_counts:
                if gt_comp_correct_port_counts.get(comp_idx, 0) == gt_comp_port_counts[comp_idx]:
                    strict_object_count += 1

        # Compute F1 scores
        def compute_f1(tp, fp, fn):
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            return precision, recall, f1

        comp_prec, comp_rec, comp_f1 = compute_f1(component_tp, component_fp, component_fn)
        port_prec, port_rec, port_f1 = compute_f1(port_tp, port_fp, port_fn)
        link_prec, link_rec, link_f1 = compute_f1(link_tp, link_fp, link_fn)

        assoc_acc = correct_associations / max(link_tp, 1) if link_tp > 0 else 0.0
        angle_mae = np.mean(angle_errors) if angle_errors else 0.0
        endpoint_mae = np.mean(endpoint_errors) if endpoint_errors else 0.0
        strict_recall = strict_object_count / max(total_objects, 1) if total_objects > 0 else 0.0

        return {
            "component_8px_f1": comp_f1,
            "port_8px_f1": port_f1,
            "link_8px_f1": link_f1,
            "link_association_accuracy": assoc_acc,
            "angle_mae": angle_mae,
            "endpoint_mae": endpoint_mae,
            "strict_object_recall": strict_recall,
        }


class StructureValidator(DetectionValidator):
    """Validator for structure models."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize structure validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "structure"
        self.metrics = StructureMetrics()
        self.kpt_shape = None
        self.component_nms_radius = 16.0
        self.port_nms_radius = 8.0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess images and keypoints."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    @staticmethod
    def _head_module(model: torch.nn.Module) -> torch.nn.Module:
        """Return the terminal structure head from wrapped validation models."""
        current = model
        for _ in range(5):
            if isinstance(current, torch.nn.Sequential):
                return current[-1]
            nested = getattr(current, "model", None)
            if nested is None or nested is current:
                break
            current = nested
        return current

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize validation metrics."""
        self.names = model.names
        self.nc = len(model.names)
        self.kpt_shape = self.data.get("kpt_shape", [25, 3])
        head = self._head_module(model)
        self.args.conf = max(float(self.args.conf or 0.0), 0.05) if head.__class__.__name__ == "StructureHeatmap" else self.args.conf
        self.metrics.reset()
        self.seen = 0

    def postprocess(self, preds: dict) -> list[dict]:
        """Filter structure predictions with confidence and radius NMS."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        outputs = []
        device = preds["components"].device

        for b in range(preds["components"].shape[0]):
            components = preds["components"][b]
            ports = preds["ports"][b]

            # Filter by confidence
            comp_keep = components[:, 2] >= self.args.conf
            components = components[comp_keep]
            port_keep = ports[:, 2] >= self.args.conf
            ports = ports[port_keep]

            # NMS
            if components.shape[0]:
                components = radius_point_nms(components, self.component_nms_radius, self.args.max_det)
            if ports.shape[0]:
                ports = radius_point_nms(ports, self.port_nms_radius, self.args.max_det)

            # Build links by matching port-predicted-component to nearest detected component
            links = []
            if ports.shape[0] and components.shape[0]:
                port_pred_comp = ports[:, 3:5]
                comp_xy = components[:, :2]
                dists = torch.cdist(port_pred_comp, comp_xy)

                for port_idx in range(ports.shape[0]):
                    min_dist, comp_idx = dists[port_idx].min(dim=0)
                    # Tolerance: min(8 cells, max(2 cells, 0.15 * distance))
                    tolerance = min(8.0 * 4.0, max(2.0 * 4.0, 0.15 * float(torch.norm(ports[port_idx, 3:5] - ports[port_idx, :2]))))
                    if min_dist <= tolerance:
                        link_score = torch.sqrt(ports[port_idx, 2] * components[comp_idx, 2])
                        link_score *= torch.exp(-0.5 * (min_dist / tolerance) ** 2)
                        links.append(torch.tensor([port_idx, comp_idx, link_score, ports[port_idx, 5], ports[port_idx, 6], ports[port_idx, 3], ports[port_idx, 4]], device=device))

            if links:
                links = torch.stack(links)
            else:
                links = torch.zeros((0, 7), device=device)

            outputs.append({
                "components": components,
                "ports": ports,
                "links": links,
            })
        return outputs

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare ground-truth for a batch image."""
        idx = batch["batch_idx"] == si
        kpts = batch["keypoints"][idx].clone()
        h, w = batch["img"].shape[2:]
        imgsz = torch.tensor((h, w), device=kpts.device, dtype=kpts.dtype)
        ori_shape = batch["ori_shape"][si]
        ratio_pad = batch["ratio_pad"][si]

        # Extract components (keypoint 0) and ports (keypoints 1-24)
        gt_components = []
        gt_ports = []
        gt_links = []

        for i, obj_kpts in enumerate(kpts):
            # Component center is first keypoint
            comp_kpt = obj_kpts[0]
            if comp_kpt[2] > 0:
                comp_xy = comp_kpt[:2].clone()
                comp_xy[0] *= w
                comp_xy[1] *= h
                comp_xy = ops.scale_coords(batch["img"].shape[2:], comp_xy.view(1, 2), ori_shape, ratio_pad=ratio_pad).view(2)
                gt_components.append(torch.cat((comp_xy, torch.tensor([1.0], device=comp_xy.device))))
                comp_idx = len(gt_components) - 1

                # Ports are keypoints 1-24
                for port_offset, port_kpt in enumerate(obj_kpts[1:]):
                    if port_kpt[2] > 0:
                        port_xy = port_kpt[:2].clone()
                        port_xy[0] *= w
                        port_xy[1] *= h
                        port_xy = ops.scale_coords(batch["img"].shape[2:], port_xy.view(1, 2), ori_shape, ratio_pad=ratio_pad).view(2)
                        gt_ports.append(torch.cat((port_xy, torch.tensor([1.0], device=port_xy.device))))
                        port_idx = len(gt_ports) - 1
                        gt_links.append(torch.tensor([port_idx, comp_idx, 1.0], device=port_xy.device))

        if gt_components:
            gt_components = torch.stack(gt_components)
        else:
            gt_components = torch.zeros((0, 3), device=kpts.device)
        if gt_ports:
            gt_ports = torch.stack(gt_ports)
        else:
            gt_ports = torch.zeros((0, 3), device=kpts.device)
        if gt_links:
            gt_links = torch.stack(gt_links)
        else:
            gt_links = torch.zeros((0, 3), device=kpts.device)

        return {
            "components": gt_components,
            "ports": gt_ports,
            "links": gt_links,
            "ori_shape": ori_shape,
            "imgsz": batch["img"].shape[2:],
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def update_metrics(self, preds: list[dict], batch: dict[str, Any]) -> None:
        """Update structure metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            gt = self._prepare_batch(si, batch)

            # Scale predictions to original image coords
            pred_components = pred["components"]
            pred_ports = pred["ports"]
            pred_links = pred["links"]

            if pred_components.numel():
                pred_comp_xy = pred_components[:, :2].clone()
                pred_comp_xy = ops.scale_coords(batch["img"].shape[2:], pred_comp_xy, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                pred_components_scaled = torch.cat((pred_comp_xy, pred_components[:, 2:]), -1)
            else:
                pred_components_scaled = pred_components.new_zeros((0, 3))

            if pred_ports.numel():
                pred_port_xy = pred_ports[:, :2].clone()
                pred_port_xy = ops.scale_coords(batch["img"].shape[2:], pred_port_xy, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                # Also scale predicted component endpoints
                pred_port_predcomp = pred_ports[:, 3:5].clone()
                pred_port_predcomp = ops.scale_coords(batch["img"].shape[2:], pred_port_predcomp, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                pred_ports_scaled = torch.cat((pred_port_xy, pred_ports[:, 2:3], pred_port_predcomp, pred_ports[:, 5:]), -1)
            else:
                pred_ports_scaled = pred_ports.new_zeros((0, 8))

            # Links reference component/port indices which don't need scaling
            self.metrics.update(pred_components_scaled, pred_ports_scaled, pred_links, gt["components"], gt["ports"], gt["links"])

    def get_stats(self) -> dict[str, Any]:
        """Return validation statistics."""
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """Gather metrics from all DDP ranks."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_records = []
            merged_nc = 0
            merged_np = 0
            merged_nl = 0
            for s in gathered_stats:
                merged_records.extend(s["records"])
                merged_nc += s["num_gt_components"]
                merged_np += s["num_gt_ports"]
                merged_nl += s["num_gt_links"]
            self.metrics.records = merged_records
            self.metrics.num_gt_components = merged_nc
            self.metrics.num_gt_ports = merged_np
            self.metrics.num_gt_links = merged_nl
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.seen = len(self.dataloader.dataset)
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            self.metrics.clear_stats()
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []

    def get_desc(self) -> str:
        """Return validation progress header."""
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Comp_F1",
            "Port_F1",
            "Link_F1",
            "Assoc_Acc",
            "Angle_MAE",
            "Endpt_MAE",
            "Strict_Rec",
        )

    def print_results(self) -> None:
        """Print aggregate structure metrics."""
        comp_f1, port_f1, link_f1, assoc_acc, angle_mae, endpt_mae, strict_rec, _ = self.metrics.mean_results()
        LOGGER.info(
            ("%22s" + "%11i" + "%11.3g" * 7) % (
                "all",
                self.seen,
                comp_f1,
                port_f1,
                link_f1,
                assoc_acc,
                angle_mae,
                endpt_mae,
                strict_rec,
            )
        )

    def finalize_metrics(self) -> None:
        """Attach speed and save_dir to metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
```

### Step 4.5: Create __init__.py
```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import StructurePredictor
from .train import StructureTrainer
from .val import StructureValidator

__all__ = "StructurePredictor", "StructureTrainer", "StructureValidator"
```

### Step 4.6: Commit
```bash
cd /data-ssd/libo/ultralytics
mkdir -p ultralytics/models/yolo/structure
git add ultralytics/models/yolo/structure/__init__.py
git add ultralytics/models/yolo/structure/train.py
git add ultralytics/models/yolo/structure/val.py
git add ultralytics/models/yolo/structure/predict.py
git add ultralytics/models/yolo/structure/utils.py
git commit -m "feat: Add structure task trainer/validator/predictor"
```

---

## Task 5: Update Model Registry and Task Mapping

**Files:**
- Modify: `/data-ssd/libo/ultralytics/ultralytics/models/yolo/__init__.py`
- Modify: `/data-ssd/libo/ultralytics/ultralytics/engine/trainer.py` (or wherever task mapping is)

### Step 5.1: Update yolo/__init__.py
Add to the imports:
```python
from ultralytics.models.yolo import structure
```

And update the `TASK_MAP` to include:
```python
    "structure": {
        "model": StructureModel,
        "trainer": structure.StructureTrainer,
        "validator": structure.StructureValidator,
        "predictor": structure.StructurePredictor,
    },
```

### Step 5.2: Commit
```bash
cd /data-ssd/libo/ultralytics
git add ultralytics/models/yolo/__init__.py
git commit -m "feat: Register structure task in task map"
```

---

## Task 6: Create YAML Model Configuration

**Files:**
- Create: `/data-ssd/libo/ultralytics/ultralytics/cfg/models/11/yolo11n-structure.yaml`

### Step 6.1: Write YAML config
```yaml
# Ultralytics YOLO11 structure prediction model
# Task: structure - predicts circuit components, ports, and their polar relationships

nc: 1  # number of classes (single-class for this task)
kpt_shape: [25, 3]  # 1 component center + 24 generic ports, with visibility flag
scales: # model compound scaling constants
  n: [0.33, 0.25, 1024]

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 15 (P2/4-small)

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, False]] # 18 (P1/2-XSmall)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, False]] # 21 (P2/4)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 24 (P3/8)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 27 (P4/16)

  - [[18, 21, 24, 27], 1, StructureHeatmap, [25, 3, 128, 384]] # [kpt_shape, topk_component, topk_port]
```

### Step 6.2: Commit
```bash
cd /data-ssd/libo/ultralytics
mkdir -p ultralytics/cfg/models/11
git add ultralytics/cfg/models/11/yolo11n-structure.yaml
git commit -m "feat: Add yolo11n-structure.yaml config"
```

---

## Task 7: Create Dataset Converter and YAML

**Files:**
- Create: `/data-ssd/libo/ultralytics/ultralytics/data/convert_device_ports.py`
- Create: `/data-ssd/libo/ultralytics/datasets/device_ports.yaml`

### Step 7.1: Write converter script
```python
"""Convert device_ports dataset to YOLO structure format."""

from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def convert_dataset(
    src_json="/data-ssd/libo/p100/StructureDetector/zujian_det/datasets/device_ports_train_4.23.json",
    dst_root="/data-ssd/libo/ultralytics/datasets/device_ports",
    train_ratio=0.9,
):
    """Convert device_ports JSON dataset to YOLO structure format."""
    dst_root = Path(dst_root)
    (dst_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    with open(src_json, "r") as f:
        data = json.load(f)

    # Split dataset
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(len(data) * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]

    def process_sample(idx, sample, split_name):
        """Process one sample and save to split."""
        if not sample.get("images"):
            return
        img_path = Path(sample["images"][0])
        if not img_path.exists():
            # Try to find in alternative locations
            alt_path = Path("/data/liuyuhao/Qwen3/train_data_260423") / img_path.name
            if alt_path.exists():
                img_path = alt_path
            else:
                return

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]

        # Copy image
        dst_img = dst_root / "images" / split_name / f"{idx:06d}.png"
        shutil.copy2(img_path, dst_img)

        # Parse annotations
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            return
        gt_text = conversations[1].get("value", "")

        # Extract JSON from GT text
        try:
            if gt_text.startswith("{"):
                gt_json = json.loads(gt_text)
            else:
                # Find JSON substring
                start = gt_text.find("{")
                end = gt_text.rfind("}")
                if start >= 0 and end > start:
                    gt_json = json.loads(gt_text[start:end+1])
                else:
                    return
        except json.JSONDecodeError:
            return

        device_ports = gt_json.get("device_ports", {})

        # Create labels: one object per component, with keypoints [center, port1, port2, ...]
        labels = []
        for comp_id, comp_data in device_ports.items():
            # Component bbox (optional, used for Gaussian radius)
            bbox = comp_data.get("bbox", [])
            if len(bbox) == 4 and bbox[0] == "top_left":
                # Format: ["top_left", [x1, y1], "bottom_right", [x2, y2]]
                x1, y1 = bbox[1]
                x2, y2 = bbox[3]
                # Convert to normalized xywh
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bx = (x1 + x2) / 2 / w
                by = (y1 + y2) / 2 / h
            else:
                bx, by, bw, bh = 0.5, 0.5, 0.0, 0.0  # dummy if no bbox

            # Initialize keypoints: [25, 3] = 1 center + 24 ports
            keypoints = np.zeros((25, 3), dtype=np.float32)

            # Component center (compute from bbox if available, or use port centroid)
            ports = comp_data.get("Ports", {})
            port_coords = []
            for port_name, port_xy in ports.items():
                if len(port_xy) == 2:
                    port_coords.append(port_xy)

            if len(port_coords) > 0:
                comp_center = np.mean(port_coords, axis=0)
            elif len(bbox) == 4:
                comp_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            else:
                continue  # skip if no info

            # Component center as keypoint 0
            keypoints[0, 0] = comp_center[0] / w  # normalized x
            keypoints[0, 1] = comp_center[1] / h  # normalized y
            keypoints[0, 2] = 2.0  # visible

            # Ports as keypoints 1-24
            for i, (port_name, port_xy) in enumerate(list(ports.items())[:24]):
                if len(port_xy) == 2:
                    keypoints[1 + i, 0] = port_xy[0] / w
                    keypoints[1 + i, 1] = port_xy[1] / h
                    keypoints[1 + i, 2] = 2.0  # visible

            # Format: class_id, x, y, w, h, kpt0_x, kpt0_y, kpt0_v, kpt1_x, kpt1_y, kpt1_v, ...
            label_line = [0, bx, by, bw, bh]
            for kpt in keypoints:
                label_line.extend([kpt[0], kpt[1], kpt[2]])

            labels.append(label_line)

        # Save label file
        if labels:
            dst_label = dst_root / "labels" / split_name / f"{idx:06d}.txt"
            with open(dst_label, "w") as f:
                for label_line in labels:
                    f.write(" ".join([f"{x:.6f}" for x in label_line]) + "\n")

    # Process all samples
    for i, idx in enumerate(tqdm(train_indices, desc="Converting train")):
        process_sample(idx, data[idx], "train")
    for i, idx in enumerate(tqdm(val_indices, desc="Converting val")):
        process_sample(idx, data[idx], "val")

    print(f"Converted {len(train_indices)} train, {len(val_indices)} val samples")


if __name__ == "__main__":
    convert_dataset()
```

### Step 7.2: Write dataset YAML
```yaml
# Device Ports Dataset - Circuit Structure Detection
# Single-class component and port detection with polar relationships

path: /data-ssd/libo/ultralytics/datasets/device_ports
train: images/train
val: images/val

# Classes
names:
  0: component

# Keypoint format: 25 keypoints (1 component center + 24 generic ports) with visibility
kpt_shape: [25, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```

### Step 7.3: Run converter
```bash
cd /data-ssd/libo/ultralytics
python ultralytics/data/convert_device_ports.py
```

### Step 7.4: Commit
```bash
cd /data-ssd/libo/ultralytics
git add ultralytics/data/convert_device_ports.py
git add datasets/device_ports.yaml
git commit -m "feat: Add device_ports dataset converter"
```

---

## Task 8: Test Training Pipeline

**Files:** N/A (test scripts)

### Step 8.1: Run quick test training
```bash
cd /data-ssd/libo/ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11n-structure.yaml')
model.train(data='datasets/device_ports.yaml', epochs=1, imgsz=640, batch=8, device=0)
"
```

### Step 8.2: Debug any issues
Fix any import errors, shape mismatches, etc.

### Step 8.3: Commit fixes if needed
```bash
cd /data-ssd/libo/ultralytics
# Add any fixes
git commit -m "fix: Resolve training pipeline issues"
```

---

## Plan Complete and Saved!

Plan complete and saved to `docs/superpowers/plans/2026-07-14-structure-centernet-polar.md`.

### Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch fresh subagent per task with review between tasks for fast iteration.

**2. Inline Execution** - Execute tasks in this session using superpowers/executing-plans with batch execution and checkpoints for review.

### Which approach would you like?
