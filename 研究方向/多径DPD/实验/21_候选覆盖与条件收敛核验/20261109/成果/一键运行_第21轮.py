#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第21轮 PyCharm 一键入口。右键 Run 本文件即可；无需命令行参数。

本地程序只使用已有common18的索引0/8/16；不修改公共MP或任何既有数据。
每次创建独立运行目录，先自检、保存盲决策，再离线真值归因。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(
    (p for p in HERE.resolve().parents
     if (p / "AGENTS.md").is_file() and (p / "研究方向/多径DPD").is_dir()),
    None)
if ROOT is None:
    raise FileNotFoundError("必须把本入口放在现有仓库的第21轮/成果/目录下。")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def main():
    # 本地已有解释器即可，不安装依赖、不改工作目录配置。
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = HERE.parent / "运行数据" / ("pilot3_" + now)
    out.mkdir(parents=True, exist_ok=False)
    status = {
        "name": "第21轮候选覆盖与条件收敛核验",
        "state": "STARTED",
        "scope": "ONLY_3_FROZEN_DEVELOPMENT_IQ",
        "indices": [0, 8, 16],
        "output": str(out),
        "interpreter": sys.executable,
        "entry_sha256": sha(Path(__file__)),
        "probe_sha256": sha(HERE / "round21_probe.py"),
        "test_sha256": sha(HERE / "test_round21.py"),
        "not_formal_batch": True,
    }
    write(out / "运行状态.json", status)
    try:
        print("第21轮：核验数学及Python接口...", flush=True)
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONUTF8"] = "1"
        subprocess_env["PYTHONIOENCODING"] = "utf-8:replace"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(HERE / "test_round21.py")],
            cwd=str(ROOT), env=subprocess_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", check=False)
        (out / "单元检查.log").write_text(result.stdout or "", encoding="utf-8")
        if result.returncode:
            raise RuntimeError("本地单元检查失败，详情见 单元检查.log")
        status["unit_test"] = "PASS"
        write(out / "运行状态.json", status)

        # 唯一运行入口直接使用已存的老观察数据，不生成或覆盖任何IQ。
        spec = importlib.util.spec_from_file_location(
            "r21_user_entry", HERE / "round21_probe.py")
        if spec is None or spec.loader is None:
            raise ImportError("第21轮核心源码无法载入")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not module.COMMON.joinpath("rows.json").is_file():
            status["state"] = "NEEDS_FROZEN_COMMON18"
            status["message"] = "找不到原common18文件；仅完成单元测试，没有运行新的IQ诊断。"
            write(out / "运行状态.json", status)
            print(status["message"], flush=True)
            return

        status["state"] = "RUNNING_FROZEN_IQ"
        write(out / "运行状态.json", status)
        print("开始3条冻结IQ限定诊断（先全部盲决策，后真值归因）...", flush=True)
        outcome = module.run(out / "冻结IQ诊断")
        status.update(state="R21_DIAGNOSTIC_FINISHED", outcome=outcome)
        write(out / "运行状态.json", status)
        print("第21轮诊断完成。请将本轮运行数据JSON推送GitHub供研究分析：", out, flush=True)
    except BaseException as exc:
        status["state"] = "FAILED_OR_INCOMPLETE"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write(out / "运行状态.json", status)
        print("第21轮未完成；原样保存错误、已生成盲决策及审计文件：", out, flush=True)
        raise


if __name__ == "__main__":
    main()
