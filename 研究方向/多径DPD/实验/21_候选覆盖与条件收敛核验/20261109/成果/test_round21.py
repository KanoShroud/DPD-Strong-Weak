#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第21轮启动前独立数学/接口测试，不运行任何冻结IQ定位。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError("模块不能加载：" + str(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    probe = load("r21_unit", HERE / "round21_probe.py")
    r20 = load("r20_unit", probe.R20 / "成果/branch_probe.py")
    assert probe.SPECS == ((0, "IN_100_R00_L0", (0, 4)),
                            (8, "IN_102_R00_L2_P075", (4, 16)),
                            (16, "IN_105_R00_L1_P075", (4, 8)))
    assert probe.BUDGETS == (70, 300)
    assert probe.PG_GATE == 1e-7
    a = probe.selftest()
    b = r20.selftest()
    rng = np.random.default_rng(2026110921)
    pool = rng.normal(size=(40, 2))
    t = pool[[3, 29]]
    best = probe.pair_error(pool, t)
    assert best["rmse_m"] < 1e-14
    assert probe.rmse(t[::-1], t) < 1e-14
    # 每个第21轮位置分支仍必须使用独立构建的路径列表与共同波形消元。
    source = (HERE / "round21_probe.py").read_text(encoding="utf-8")
    assert "r20.propose_path(" in source
    assert "r20.fixed_nuisance_fit(" in source
    assert "engine.objective(" in source
    assert "blind_stage_completed.json" in source
    result = {"status": "PASS", "r21_unit": a, "r20_math": b,
              "data_accessed": False,
              "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
