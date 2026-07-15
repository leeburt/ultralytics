# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ultralytics.models.yolo.structure.predict import StructurePredictor
from ultralytics.models.yolo.structure.utils import (
    associate_ports_to_components,
    decode_port_distance,
    encode_port_distance,
)
from ultralytics.models.yolo.structure.val import StructureMetrics, scale_link_geometry


class TestStructurePostprocessing(unittest.TestCase):
    def test_port_distance_encoding_round_trip(self):
        """The relation target must invert the decoder's softplus/expm1 transform."""
        distance = torch.tensor([0.5, 4.0, 12.0])

        self.assertTrue(torch.allclose(decode_port_distance(encode_port_distance(distance)), distance, atol=1e-5))

    def test_association_suppresses_same_ray_duplicate_and_keeps_distinct_port(self):
        """M1-style duplicate points on one terminal must not remove the opposite terminal."""
        components = torch.tensor([[320.5, 278.7, 0.94]])
        # xy, score, predicted component xy, direction xy, raw rho
        ports = torch.tensor([
            [348.5, 238.0, 0.624, 319.0, 280.9, -0.54, 0.84, 2.385],  # M1 upper terminal
            [348.1, 318.7, 0.658, 328.1, 269.8, -0.52, -0.85, 2.398],  # M1 lower terminal
            [348.1, 252.3, 0.188, 313.5, 279.7, -0.63, 0.78, 2.218],  # upper-terminal duplicate
        ])

        kept_ports, links = associate_ports_to_components(components, ports)

        self.assertEqual(kept_ports.shape[0], 2)
        self.assertTrue(torch.allclose(kept_ports[:, 2].sort().values, torch.tensor([0.624, 0.658])))
        self.assertEqual(links[:, :2].tolist(), [[0.0, 0.0], [1.0, 0.0]])

    def test_association_keeps_nearby_same_side_ports(self):
        """Adjacent terminals on the same side of a symbol are not duplicate heatmap peaks."""
        components = torch.tensor([[0.0, 0.0, 0.9]])
        ports = torch.tensor([
            [100.0, 0.0, 0.9, 0.0, 0.0, -1.0, 0.0, 1.0],
            [100.0, 12.0, 0.8, 0.0, 0.0, -1.0, 0.0, 1.0],
        ])

        kept_ports, links = associate_ports_to_components(components, ports)

        self.assertEqual(kept_ports.shape[0], 2)
        self.assertEqual(links.shape[0], 2)

    def test_link_fn_is_counted_per_image(self):
        """A true positive in one image must not erase a missed link in the next image."""
        metrics = StructureMetrics()
        component = torch.tensor([[10.0, 10.0, 0.9]])
        port = torch.tensor([[5.0, 10.0, 0.9]])
        link = torch.tensor([[0.0, 0.0, 0.9]])
        metrics.update(component, port, link, component, port, link)
        metrics.update(
            torch.empty((0, 3)),
            torch.empty((0, 3)),
            torch.empty((0, 3)),
            component,
            port,
            link,
        )

        self.assertAlmostEqual(metrics._compute_metrics()["link_8px_f1"], 2.0 / 3.0)

    def test_association_accuracy_counts_wrong_component_links(self):
        """A matched port linked to the other detected component is an incorrect association."""
        metrics = StructureMetrics()
        components = torch.tensor([[10.0, 10.0, 0.9], [40.0, 10.0, 0.9]])
        ports = torch.tensor([[5.0, 10.0, 0.9], [45.0, 10.0, 0.9]])
        gt_links = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        pred_links = torch.tensor([[0.0, 0.0, 0.9], [1.0, 0.0, 0.9]])
        metrics.update(components, ports, pred_links, components, ports, gt_links)

        self.assertAlmostEqual(metrics._compute_metrics()["link_association_accuracy"], 0.5)

    def test_scaled_link_geometry_uses_scaled_port_endpoint(self):
        """Endpoint and direction metrics must be evaluated in the original-image coordinate system."""
        links = torch.tensor([[0.0, 0.0, 0.9, 0.0, 1.0, 100.0, 50.0]])
        scaled_ports = torch.tensor([[100.0, 50.0, 0.9, 200.0, 50.0, 0.0, 1.0, 2.0]])

        scaled_links = scale_link_geometry(links, scaled_ports)

        self.assertTrue(torch.allclose(scaled_links[0, 3:5], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(scaled_links[0, 5:7], torch.tensor([200.0, 50.0])))

    def test_predictor_returns_structure_without_misusing_keypoints(self):
        """Structure output is not a pose tensor and must not be passed to Results.keypoints."""
        predictor = StructurePredictor(overrides={"conf": 0.1, "max_det": 10})
        predictor.batch = (["example.png"],)
        predictor.model = SimpleNamespace(names={0: "structure"})
        predictions = ([{
            "components": torch.tensor([[20.0, 32.0, 0.9]]),
            "ports": torch.tensor([[10.0, 32.0, 0.9, 20.0, 32.0, 1.0, 0.0, 1.0]]),
        }], None)

        result = predictor.postprocess(predictions, torch.zeros((1, 3, 64, 64)), [np.zeros((128, 256, 3), np.uint8)])[0]

        self.assertIsNone(result.keypoints)
        self.assertEqual(result.structure["ports"].shape, (1, 1, 3))
        self.assertEqual(result.structure["links"].shape, (1, 7))
        self.assertTrue(torch.allclose(result.structure["links"][:, 5:7], result.structure["pred_components"].squeeze(1)))
        self.assertEqual(len(result), 2)
        self.assertIn("component", result.verbose())
        self.assertTrue(hasattr(result.cpu(), "structure"))
        self.assertTrue(hasattr(result.numpy(), "structure"))
        self.assertTrue(np.any(result.plot() != 0))
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "structure.png"
            result.save(str(output_path))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
