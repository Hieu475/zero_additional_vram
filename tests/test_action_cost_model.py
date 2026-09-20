"""Tests for MeasuredActionCostModel (Phase 8)."""

from __future__ import annotations

import pytest
import numpy as np

from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.controllers.hardware_controller import HardwareAwareJointController, ControllerObservation


class TestActionCostModel:
    """Validate data-driven action cost model."""

    def test_cost_model_from_files(self):
        model = MeasuredActionCostModel.from_files()
        assert model.baseline_tps > 0
        assert len(model.pareto_results) > 0 or len(model.action_costs) > 0

    def test_latency_predictions(self):
        model = MeasuredActionCostModel.from_files()

        # Draft latency must scale with K
        t_d1 = model.predict_draft_ms("cka_83", k=1)
        t_d2 = model.predict_draft_ms("cka_83", k=2)
        t_d4 = model.predict_draft_ms("cka_83", k=4)
        assert 10.0 < t_d1 < 35.0
        assert t_d1 < t_d2 < t_d4

        # Verification latency must be positive and scale sub-linearly
        t_v1 = model.predict_verify_ms(k=1)
        t_v4 = model.predict_verify_ms(k=4)
        assert 20.0 < t_v1 < 40.0
        assert t_v1 < t_v4

    def test_entropy_modulation_on_acceptance(self):
        model = MeasuredActionCostModel.from_files()

        # Low entropy (high model certainty) must increase predicted acceptance rate
        acc_low_entropy = model.predict_acceptance_rate("cka_83", k=2, entropy=0.2)
        acc_high_entropy = model.predict_acceptance_rate("cka_83", k=2, entropy=2.5)

        assert acc_low_entropy > acc_high_entropy
        assert 0.10 <= acc_high_entropy <= 1.0
        assert 0.10 <= acc_low_entropy <= 1.0

    def test_action_evaluation_and_utility(self):
        model = MeasuredActionCostModel.from_files()

        pred = model.evaluate_action(
            config_name="cka_83",
            k=2,
            entropy=0.5,
            vram_used_mb=2000.0,
            gpu_power_w=55.0,
            gpu_temp_c=65.0,
        )

        assert pred.draft_ms > 0
        assert pred.verify_ms > 0
        assert pred.expected_tokens_per_step > 1.0
        assert pred.expected_tps > 0
        assert pred.expected_speedup > 0
        assert isinstance(pred.utility, float)

    def test_hardware_penalties(self):
        model = MeasuredActionCostModel.from_files()

        # Normal condition
        normal_eval = model.evaluate_action(
            config_name="cka_83",
            k=2,
            entropy=1.0,
            gpu_temp_c=60.0,
            vram_used_mb=2000.0,
        )

        # Thermal throttling condition (near 83C limit)
        hot_eval = model.evaluate_action(
            config_name="cka_83",
            k=2,
            entropy=1.0,
            gpu_temp_c=81.0,  # within 5C of 82C limit
            vram_used_mb=2000.0,
        )

        assert hot_eval.utility < normal_eval.utility


class TestJointControllerWithCostModel:
    """Validate joint controller using MeasuredActionCostModel."""

    def test_controller_decision_making(self):
        configs = {
            "cka_83": [4, 5, 6, 7, 12, 13],
            "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
        }
        controller = HardwareAwareJointController(candidate_layer_configs=configs)

        action = controller.select_action(
            entropy=0.4,
            last_accepted=2,
            last_proposed=2,
            draft_ms=20.0,
            verify_ms=27.0,
        )

        assert action.config_name in configs
        # Due to empirical quality dominance, CKA-83 should be preferred over CKA-50
        assert action.config_name == "cka_83"
        assert 1 <= action.draft_length <= 8
