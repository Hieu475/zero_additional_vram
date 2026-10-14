import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/run_capacity_quantization_control.py"
SPEC = importlib.util.spec_from_file_location("capacity_control", SCRIPT)
capacity_control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capacity_control)


def test_shared_prefix_logits_stop_after_first_different_choice():
    ref_tokens = [11, 12, 13]
    actual_tokens = [11, 99, 100]
    ref_logits = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5])]
    actual_logits = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.9]), torch.tensor([0.4, 0.6])]

    rows = capacity_control.compare_logits_on_shared_prefix(
        actual_logits, ref_logits, actual_tokens, ref_tokens)

    # Position 1 predicts the first differing token but still has the same input prefix.
    assert [r["generated_prefix_length"] for r in rows] == [0, 1]
    assert rows[0]["max_abs"] == 0.0
    assert rows[1]["max_abs"] > 0.0


def test_equal_tokens_compare_all_available_prefix_rows():
    logits = [torch.tensor([1.0, -1.0]), torch.tensor([2.0, 0.0])]
    rows = capacity_control.compare_logits_on_shared_prefix(logits, logits, [2, 3], [2, 3])
    assert len(rows) == 2
    assert all(r["bitwise_equal"] for r in rows)


def test_kv_comparison_reports_active_layer_errors():
    ref = [(torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4), 3)]
    actual = [(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4), 3)]

    result = capacity_control.compare_kv(actual, ref)

    assert result["logical_length_equal"]
    assert result["max_key_abs_error"] == 1.0
    assert result["max_value_abs_error"] == 0.0
    assert not result["all_active_kv_bitwise_equal"]
