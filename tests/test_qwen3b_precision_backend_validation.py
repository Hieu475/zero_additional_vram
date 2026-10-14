import json

import torch

from scripts.run_qwen3b_precision_backend_validation import errors, select_rows


def test_select_rows_labels_diagnostic_and_heldout(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    config = tmp_path / "config.json"
    rows = [{"prompt_id": f"p{i}", "input_ids": [i, i + 1]} for i in range(5)]
    inputs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    config.write_text(json.dumps({"ids": ["p3", "p1"]}), encoding="utf-8")

    selected = select_rows(inputs, config, diagnostic_n=2, heldout_n=2, ids_override=None)

    assert [(row["prompt_id"], cohort) for row, cohort in selected] == [
        ("p3", "diagnostic"), ("p1", "diagnostic"),
        ("p0", "heldout"), ("p2", "heldout"),
    ]


def test_logit_error_metrics_distinguish_exact_and_drift():
    reference = torch.tensor([1.0, 2.0, 3.0])
    exact = errors(reference, reference.clone())
    drifted = errors(torch.tensor([1.0, 2.0, 3.1]), reference)

    assert exact["bitwise_equal"] is True
    assert exact["max_abs"] == 0.0
    assert drifted["bitwise_equal"] is False
    assert drifted["max_abs"] > 0
    assert drifted["max_relative"] > 0
