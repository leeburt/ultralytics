# Structure CenterNet Polar Head Design

## Status

Approved architecture discussion. This document defines the implementation
target before a feature branch or training code is created.

## Scope

Build a new Ultralytics task for single-class circuit structure prediction:

- one `COMPONENT` center heatmap;
- one generic `PORT` heatmap;
- a `PORT -> COMPONENT` relation represented by direction and distance;
- a structure-aware decoder and validator.

The implementation will be based on `/data-ssd/libo/ultralytics` branch
`exp/keypoint-centernet-strict-loss`, not on
`/data-ssd/libo/p100/ultralytics`. The latter is an older upstream checkout
without the CenterNet keypoint branch.

The initial experiment intentionally uses a single `COMPONENT` class and a
single `PORT` class. Multi-class component and port prediction are out of
scope for this first experiment.

## Rationale

The existing StructureDetector model predicts component and port heatmaps,
but shares one offset map between them and regresses a Cartesian port-to-center
vector. The shared offset has conflicting supervision when different target
types occupy the same feature cell. The Cartesian vector mixes direction and
distance on an unbounded scale.

The selected design keeps CenterNet's dense point formulation, separates the
two detection tasks, and represents the association vector in polar form:

```text
direction = (cos(theta), sin(theta))
distance  = log(1 + ||component_center - port|| / stride)
```

An angle-only relation cannot determine the parent component in a crowded
circuit. Distance is therefore part of the relation head, not an optional
post-processing heuristic.

## Data Contract

The source annotations provide an axis-aligned bbox and a variable-size `Ports`
dictionary for every component. The current StructureDetector conversion drops
the bbox; the new converter must retain it.

Each component becomes one standard YOLO keypoint label row:

```text
class bbox_xywh center_x center_y center_v port_1_x port_1_y port_1_v ... port_24_x port_24_y port_24_v
```

- `class` is always `0`.
- `kpt_shape` is `[25, 3]`: slot 0 is the component center and slots 1-24
  are generic ports.
- Missing port slots use zero visibility.
- The source corpus has at most 24 ports per component, so this format does
  not truncate valid ports.
- Port slot order has no semantic meaning. The target builder pools every
  visible port slot into one generic port heatmap.
- The dataset YAML uses an identity `flip_idx` for all 25 slots. Since slots
  are generic, no left-right semantic permutation is required.

Targets are generated after Ultralytics geometric augmentation from transformed
bboxes and keypoints. This makes crop, resize, mosaic, affine transforms, and
horizontal or vertical flips consistent without separately transforming an
angle label.

## Model Architecture

Reuse the proven YOLO11 P2/P3/P4/P5 fusion pattern from
`KeypointHeatmap`, with the P2 feature as the stride-4 output grid. Replace its
terminal head with `StructureHeatmap`:

```text
P2/P3/P4/P5 -> lateral projection and sum -> shared refinement feature
                                              |
                 +----------------------------+---------------------------+
                 |                            |                           |
          component tower                 port tower                relation tower
          hm: 1, off: 2                   hm: 1, off: 2             dir: 2, rho: 1
```

All output maps have spatial shape `B x C x H/4 x W/4`. The model has nine
output channels in total:

| Map | Channels | Supervised positions |
| --- | ---: | --- |
| `component_hm` | 1 | dense heatmap |
| `component_off` | 2 | component center cells |
| `port_hm` | 1 | dense heatmap |
| `port_off` | 2 | port cells |
| `direction` | 2 | valid port cells |
| `rho` | 1 | valid port cells |

There is no predicted `wh` branch in the first experiment. Ground-truth bbox
size is used only to choose a component heatmap radius; inference and
component-port association do not consume a predicted box.

## Target Construction

Let `s = 4` be the output stride. After augmentation, convert all coordinates
to feature-cell units. Let `c` be one component center, `p` one visible port,
and `b = (w, h)` its component bbox.

1. Component heatmap:
   - draw a CenterNet Gaussian centered at `c`;
   - use `r_component = clamp(gaussian_radius(h / s, w / s), 1, 8)`;
   - use the maximum when Gaussians overlap.
2. Port heatmap:
   - draw a Gaussian centered at each `p`;
   - use fixed radius `r_port = 1` output cell to avoid broad, merging peaks
     in dense port layouts.
3. Offsets:
   - at `floor(c)`, store `c - floor(c)` in `component_off_target`;
   - at `floor(p)`, store `p - floor(p)` in `port_off_target`.
4. Relation target at the port cell:

```text
delta       = c - p
distance_gt = ||delta||_2
direction_gt = delta / distance_gt
rho_gt      = log(1 + distance_gt)
```

`distance_gt < 0.5` feature cell has no stable direction and is excluded from
relation losses. It remains a normal port detection target.

Two targets of the same type in one feature cell cannot be decoded as distinct
points at stride 4. The heatmap remains positive, while offset and relation
regression retain one deterministic target and increment a collision counter.
The converter and validator report these collisions; they are never silently
removed from data statistics.

## Losses

Let `N_component`, `N_port`, and `N_relation` be the counts of their masked
positive positions in a batch. Every loss is normalized by its own count.

### Heatmap Losses

Use the CenterNet modified focal loss separately for component and port logits:

```text
L_hm = -1 / max(N, 1) * [
  sum_{Y=1} (1-sigmoid(X))^2 * log(sigmoid(X))
  + sum_{Y<1} (1-Y)^4 * sigmoid(X)^2 * log(1-sigmoid(X))
]
```

Separate normalization is required because images contain substantially more
ports than components.

### Offset Losses

Use direct masked L1 at target cells. Do not apply sigmoid to offsets:

```text
L_component_off = sum(M_component * abs(off_hat - off_gt)) / (2 * max(N_component, 1))
L_port_off      = sum(M_port * abs(off_hat - off_gt)) / (2 * max(N_port, 1))
```

### Polar Relation Losses

For relation output `v` and raw distance output `rho_raw`:

```text
u_hat       = v / (||v||_2 + epsilon)
rho_hat     = softplus(rho_raw)
delta_hat   = expm1(rho_hat) * u_hat
```

Use three complementary masked terms:

```text
L_dir      = mean(1 - dot(u_hat, direction_gt))
L_rho      = SmoothL1(rho_hat, rho_gt)
L_endpoint = SmoothL1((delta_hat - delta) / max(distance_gt, 1))
```

`L_dir` handles the circular angle domain without a discontinuity at `-pi/pi`.
`L_rho` keeps long relation vectors numerically controlled. `L_endpoint`
couples the two outputs so individually plausible angle and distance estimates
must reconstruct the same component center.

The initial total loss is:

```text
L_total = L_component_hm + L_port_hm
        + L_component_off + L_port_off
        + 0.50 * L_dir + 0.50 * L_rho + 0.25 * L_endpoint
```

During the first five epochs, linearly ramp the three relation weights from
zero to their configured values. This lets the newly initialized relation
tower avoid destabilizing transferred backbone and detection features.

Do not use learned uncertainty weighting or GradNorm in the first experiment.
Log each raw loss and its gradient norm, then adjust static weights only if
one head demonstrably dominates.

## Decoding

1. Apply sigmoid and independent `3 x 3` local-maximum NMS to the component
   and port heatmaps.
2. Select independent top-k candidates (`K_component=128`, `K_port=384`
   initially), then filter them with head-specific confidence thresholds.
3. Gather the corresponding offset maps to recover sub-pixel point locations.
4. At each surviving port location, gather `direction` and `rho`, then compute:

```text
component_endpoint_hat = port_hat + s * expm1(rho_hat) * u_hat
```

5. Match the port to the nearest surviving component candidate. Accept only if
   its endpoint residual is below:

```text
tolerance = min(8, max(2, 0.15 * predicted_distance)) feature cells
```

6. Give an accepted association a joint score:

```text
link_score = sqrt(port_score * component_score)
             * exp(-0.5 * (endpoint_residual / tolerance)^2)
```

One port has at most one parent; a component has any number of ports. Ports
that fail association remain available in raw output for detector diagnostics,
but do not enter the assembled structure result.

Apply an optional radius NMS after local maxima independently for components
and ports. Its radii, confidence thresholds, and top-k limits are calibrated
on validation data and are never shared between the two heads.

## Evaluation

All matching occurs after inverse-letterbox conversion to original-image
coordinates. The main point tolerance is two feature cells (8 px at the
configured input scale); also report 5 px, 8 px, and 12 px sensitivity.

| Metric | Definition |
| --- | --- |
| Component F1 | Score-ordered one-to-one component center matching |
| Raw port F1 | Score-ordered one-to-one port matching without parent checks |
| Link F1 | Port geometry matches and its assigned parent maps to the same GT component |
| Association accuracy | Correct parents among geometrically matched ports |
| Angle MAE | Circular direction error for correctly matched links |
| Endpoint MAE | Predicted relation endpoint error to the GT component center |
| Strict object recall | Matched component whose complete GT port set is correctly linked |

Use a fixed decoder configuration for checkpoint selection. A confidence sweep
is diagnostic only and must not select a checkpoint by its best possible
per-epoch threshold.

```text
fitness = 0.30 * component_F1@8px + 0.70 * link_F1@8px
```

Save `best.pt` by `fitness`, plus `best_link.pt` and `best_loss.pt`. Store the
head schema, class maps, keypoint layout, decoder thresholds, epoch,
optimizer/scheduler state, and all best metrics in checkpoints. Do not retain
the obsolete bean/maize classification metric.

## Validation and Failure Handling

- A no-port component contributes to component detection but no relation loss.
- A batch with no positive targets yields a differentiable zero for sparse
  regression terms and focal negative supervision for heatmaps.
- Targets outside an augmented image are marked invisible before encoding and
  counted in augmentation statistics.
- All target capacity limits, dropped labels, and same-cell collisions are
  reported per split.
- The validator reports raw detection and assembled structure metrics together;
  raw port recall alone must not decide the best checkpoint.

## Implementation Boundaries

The implementation creates a new `structure` task with a dedicated model,
loss, predictor, validator, and exporter path. It will not change the existing
`keypoint` task or the original StructureDetector code. Initial verification
must cover target construction, loss masks and normalization, decode geometry,
augmentation consistency, export output ordering, and a short overfit run
before a full experiment is launched.
