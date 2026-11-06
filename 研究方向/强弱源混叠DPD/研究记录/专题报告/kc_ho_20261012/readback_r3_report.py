"""Verify frozen R3 rows and sample selection for manuscript use only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "outputs_e2e/unified/e2e_g9"
INPUTS = {}


def read(path):
    path = path.resolve(strict=True)
    assert path.is_relative_to(BASE)
    raw = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    out = (HERE / args.run).resolve()
    assert out.parent == HERE and not out.exists()
    directory = BASE / "r3_bound_dev_20261011_01"
    audit = read(directory / "final_audit_report.json")
    assert audit["status"] == "PASS" and audit["inputs_unchanged"]
    report = read(directory / "report.json")
    assert INPUTS[str((directory / "report.json").resolve())] == audit["report"]["sha256"]
    contract = read(directory / "contract.json")
    rows = [read(p) for p in sorted((directory / "rows").glob("*.json"))]
    selected = read(BASE / "r1_failure_20261011_01/contract.json")["selected"]
    assert len(rows) == len(selected) == 24
    assert {r["tag"] for r in rows} == set(selected)
    before = []
    for row in rows:
        assert selected[row["tag"]] == row["selection"]
        old = read(BASE / "r1_failure_20261011_01/score_rows" / f"{row['tag']}.json")
        before.append(dict(tag=row["tag"], selection=row["selection"],
                           weak_error_m=old["baseline_metrics"]["errors_m"][1]))
        truth = np.asarray(old["truth"])[1]
        for m, result in row["results"].items():
            err = np.linalg.norm(np.asarray(result["position"])-truth)
            assert abs(err-result["error_m"]) < 1e-9
            assert result["score"] >= result["initial_score"]-1e-12
            assert abs(result["initial_score"]-old["estimated"]["returned_score"]) < 1e-10
    summary = {}
    for m in contract["methods"]:
        values = [r["results"][m] for r in rows]
        s = dict(cases=len(rows), success20=sum(v["error_m"] <= 20 for v in values),
            failure_recovered20=sum(r["results"][m]["error_m"] <= 20 for r in rows if r["selection"] == "failure"),
            controls20=sum(r["results"][m]["error_m"] <= 20 for r in rows if r["selection"] == "success_control"),
            seconds_mean=float(np.mean([v["seconds"] for v in values])),
            grid_points_mean=float(np.mean([v["grid_points"] for v in values])),
            pruned_cells_mean=float(np.mean([v["pruned_cells"] for v in values])))
        for k, v in s.items():
            assert abs(v-report["summary"][m][k]) < 1e-9, (m, k)
        summary[m] = s
    controls = [b for b in before if b["selection"] == "success_control"]
    failures = [b for b in before if b["selection"] == "failure"]
    assert len(controls) == 8 and len(failures) == 16
    assert sum(b["weak_error_m"] <= 20 for b in controls) == 6
    assert all(b["weak_error_m"] > 100 for b in failures)
    initial_success_tags = {b["tag"] for b in controls if b["weak_error_m"] <= 20}
    assert all(r["results"][m]["error_m"] <= 20
               for r in rows if r["tag"] in initial_success_tags for m in ("S", "E"))
    assert all(r["results"]["S"]["grid_points"] == 16384 for r in rows)
    assert all(r["results"]["E"]["pruned_cells"] == 0 for r in rows)
    assert all(r["results"]["E"]["grid_points"] == 16384 for r in rows)
    smoke = BASE / "r3_bound_smoke_20261011_01"
    assert read(smoke / "final_audit_report.json")["status"] == "PASS"
    numerical = [v for p in sorted(smoke.glob("numerical_N*.json")) for v in read(p)]
    assert len(numerical) == 24
    assert all(v["score_error"] < 1e-12 and v["upper_excess"] < 1e-10 for v in numerical)
    for p, digest in INPUTS.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    out.mkdir()
    result = dict(status="PASS", evidence_status=report["status"], summary=summary,
        initial_controls20=6, historical_controls=8, historical_failures=16,
        initial_control_errors_m=[b["weak_error_m"] for b in controls],
        shifted_to_dense_time_ratio=summary["shift10"]["seconds_mean"]/summary["D"]["seconds_mean"],
        bound_to_dense_time_ratio=summary["E"]["seconds_mean"]/summary["D"]["seconds_mean"],
        numerical_records_verified=24, row_count=24, methods_per_row=4,
        original_20m_controls_retained_by_subdivision=True,
        scope="fixed selected cases; one weak-source update with the strong estimate held fixed",
        no_inference=True, no_training=True, inputs=INPUTS)
    (out / "readback.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "inputs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
