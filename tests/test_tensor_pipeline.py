from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


try:
    import torch

    from urbantales_ml.baselines import PatchNearestNeighborModel, geometry_condition_features
    from urbantales_ml.catalog import UrbanTalesCatalog
    from urbantales_ml.data import (
        UrbanTalesPatchDataset,
        ablate_model_input_channels,
        build_input_patch,
        load_geometry,
        scale_target,
        unscale_target,
    )
    from urbantales_ml.inference import _periodic_add_shared_weight
    from urbantales_ml.metrics import batch_metrics
    from urbantales_ml.models import FNO2d, GeoMultiScaleOperator, LightCNN, MultiTaskUNet
except ImportError:  # lets the stdlib-only split tests run before dependencies exist
    torch = None


@unittest.skipIf(torch is None, "PyTorch data dependencies are not installed")
class TensorPipelineTests(unittest.TestCase):
    def _catalog_or_skip(self) -> UrbanTalesCatalog:
        required = (
            ROOT / "metadata.csv",
            ROOT / "reports/data_audit/audit_summary.json",
            ROOT / "reports/literature_review/case_name_map.csv",
        )
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            self.skipTest(
                "UrbanTALES integration metadata are not present: " + ", ".join(missing)
            )
        return UrbanTalesCatalog(ROOT)

    def test_model_input_ablation_zeros_only_requested_channel(self) -> None:
        inputs = torch.randn(2, 6, 8, 8)
        ablated = ablate_model_input_channels(inputs, ["sdf"])
        self.assertTrue(torch.equal(ablated[:, 2], torch.zeros_like(ablated[:, 2])))
        self.assertTrue(torch.equal(ablated[:, :2], inputs[:, :2]))
        self.assertTrue(torch.equal(ablated[:, 3:], inputs[:, 3:]))
        self.assertFalse(ablated.data_ptr() == inputs.data_ptr())
        with self.assertRaisesRegex(ValueError, "Unknown model input channels"):
            ablate_model_input_channels(inputs, ["not_a_channel"])

    def test_scale_round_trip(self) -> None:
        values = torch.tensor([1.0, -2.0]).numpy()
        for target in ("uped", "TKEped"):
            restored = unscale_target(scale_target(values, target, 0.2), target, 0.2)
            self.assertTrue(torch.allclose(torch.from_numpy(restored), torch.from_numpy(values)))

    def test_real_case_to_model_output(self) -> None:
        catalog = self._catalog_or_skip()
        manifest = json.loads(
            (ROOT / "configs/splits/splits_v1.json").read_text(encoding="utf-8")
        )
        case = catalog.get(manifest["protocols"]["geometry_grouped"]["train"][0])
        dataset = UrbanTalesPatchDataset(
            ROOT,
            [case],
            patch_size_m=32,
            output_pixels=32,
            patches_per_case=1,
            random_patches=False,
            max_cache_cases=1,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["input"].shape), (6, 32, 32))
        self.assertEqual(tuple(sample["target"].shape), (4, 32, 32))
        model = MultiTaskUNet(
            in_channels=6,
            task_names=("uped", "vped", "Uped", "TKEped"),
            base_channels=8,
            depth=3,
            predict_uncertainty=True,
        )
        output = model(sample["input"].unsqueeze(0))
        self.assertEqual(tuple(output["mean"].shape), (1, 4, 32, 32))
        self.assertEqual(tuple(output["log_scale"].shape), (1, 4, 32, 32))

    def test_deterministic_evaluation_patches_are_distinct(self) -> None:
        catalog = self._catalog_or_skip()
        manifest = json.loads(
            (ROOT / "configs/splits/splits_v1.json").read_text(encoding="utf-8")
        )
        case = catalog.get(manifest["protocols"]["geometry_grouped"]["val"][0])
        dataset = UrbanTalesPatchDataset(
            ROOT,
            [case],
            patch_size_m=32,
            output_pixels=32,
            patches_per_case=2,
            random_patches=False,
            max_cache_cases=1,
        )
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first["patch_index"], 0)
        self.assertEqual(second["patch_index"], 1)
        self.assertNotEqual(first["patch_origin_yx"], second["patch_origin_yx"])

    def test_target_free_input_builder_matches_training_builder(self) -> None:
        catalog = self._catalog_or_skip()
        manifest = json.loads(
            (ROOT / "configs/splits/splits_v1.json").read_text(encoding="utf-8")
        )
        case = catalog.get(manifest["protocols"]["geometry_grouped"]["train"][0])
        dataset = UrbanTalesPatchDataset(
            ROOT,
            [case],
            patch_size_m=32,
            output_pixels=32,
            patches_per_case=1,
            random_patches=False,
            max_cache_cases=1,
        )
        sample = dataset[0]
        y0, x0 = sample["patch_origin_yx"]
        target_free = build_input_patch(
            case,
            load_geometry(ROOT, case),
            y0=y0,
            x0=x0,
            patch_size_m=32,
            output_pixels=32,
            height_scale_m=50,
            sdf_scale_m=64,
            u_tau_reference_m_s=0.21,
        )
        self.assertTrue(torch.equal(target_free, sample["input"]))

    def test_all_model_families_have_common_contract(self) -> None:
        inputs = torch.randn(1, 6, 32, 32)
        tasks = ("uped", "vped", "Uped", "TKEped")
        models = (
            LightCNN(6, tasks, width=8, layers=2, predict_uncertainty=False),
            MultiTaskUNet(6, tasks, base_channels=8, depth=2, predict_uncertainty=True),
            FNO2d(6, tasks, width=8, layers=2, modes_y=4, modes_x=4),
            GeoMultiScaleOperator(
                6,
                tasks,
                base_channels=8,
                depth=2,
                operator_levels=2,
                modes=4,
                predict_uncertainty=True,
            ),
        )
        for model in models:
            output = model(inputs)
            self.assertEqual(tuple(output["mean"].shape), (1, 4, 32, 32))

    def test_spectral_models_use_real_optimizer_parameters(self) -> None:
        """AMP GradScaler cannot update parameters registered as ComplexFloat."""
        inputs = torch.randn(1, 6, 32, 32)
        tasks = ("uped", "vped", "Uped", "TKEped")
        models = (
            FNO2d(6, tasks, width=8, layers=1, modes_y=4, modes_x=4),
            GeoMultiScaleOperator(
                6,
                tasks,
                base_channels=8,
                depth=2,
                operator_levels=2,
                modes=4,
                predict_uncertainty=True,
            ),
        )
        for model in models:
            self.assertFalse(any(parameter.is_complex() for parameter in model.parameters()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            model(inputs)["mean"].square().mean().backward()
            optimizer.step()

    def test_physical_and_regional_metrics_use_target_scales(self) -> None:
        tasks = ("uped", "vped", "Uped", "TKEped")
        prediction = torch.ones(1, 4, 4, 4)
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
        inputs = torch.zeros(1, 6, 4, 4)
        inputs[:, 2] = 4.0 / 64.0  # all valid pixels are 4 m from a building
        metrics = batch_metrics(
            prediction,
            target,
            mask,
            tasks,
            u_tau_m_s=torch.tensor([0.2]),
            input_tensor=inputs,
            log_scale=torch.zeros_like(prediction),
            pixel_size_m=1.0,
            sdf_scale_m=64.0,
            near_building_distance_m=8.0,
        )
        self.assertAlmostEqual(metrics["uped/physical_m_s/mae"], 0.2, places=6)
        self.assertAlmostEqual(metrics["TKEped/physical_m2_s2/mae"], 0.04, places=6)
        self.assertAlmostEqual(metrics["uped/near_building_m_s/mae"], 0.2, places=6)
        self.assertAlmostEqual(metrics["vector/error_magnitude_mae_m_s"], 2**0.5 * 0.2, places=6)
        self.assertAlmostEqual(
            metrics["horizontal_divergence/error_mae_s_inv"], 0.0, places=6
        )
        self.assertEqual(metrics["uped/uncertainty_physical_m_s/coverage_50"], 0.0)
        self.assertEqual(metrics["uped/uncertainty_physical_m_s/coverage_90"], 1.0)

    def test_periodic_overlap_add_handles_tiles_larger_than_domain(self) -> None:
        import numpy as np

        value_sum = np.zeros((1, 2, 3), dtype=np.float32)
        weight_sum = np.zeros((2, 3), dtype=np.float32)
        values = np.ones((1, 4, 5), dtype=np.float32)
        weights = np.ones((4, 5), dtype=np.float32)
        _periodic_add_shared_weight(value_sum, weight_sum, values, weights, 1, 2)
        self.assertTrue(np.all(weight_sum > 0))
        self.assertTrue(np.allclose(value_sum[0] / weight_sum, 1.0))

    def test_geometry_nearest_neighbor_uses_auditable_training_donors(self) -> None:
        bank_inputs = torch.stack(
            (torch.zeros(6, 8, 8), torch.ones(6, 8, 8))
        )
        bank_geometry, bank_conditions = geometry_condition_features(bank_inputs, 2)
        bank_targets = torch.stack(
            (torch.full((1, 8, 8), 3.0), torch.full((1, 8, 8), 7.0))
        )
        model = PatchNearestNeighborModel(
            bank_geometry,
            bank_conditions,
            bank_targets,
            ("train-zero", "train-one"),
            embedding_pixels=2,
            boundary_weight=1.0,
        )
        output = model(torch.stack((torch.zeros(6, 8, 8), torch.ones(6, 8, 8))))
        self.assertTrue(torch.equal(output["mean"], bank_targets))
        self.assertEqual(model.match_counts(), {"train-one": 1, "train-zero": 1})


if __name__ == "__main__":
    unittest.main()
