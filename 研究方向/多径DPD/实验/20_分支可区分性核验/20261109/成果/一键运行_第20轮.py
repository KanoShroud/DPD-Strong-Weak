#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第20轮｜PyCharm 一键运行入口。

直接右键本文件 -> Run，不需要配置工作目录、命令行参数或环境变量。
先运行算法自检和独立合成单元检查，然后执行预登记的三条冻结开发 IQ。
不会改动旧轮次源码和数据；不会自动启动18条或正式大批实验。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for parent in HERE.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "研究方向" / "多径DPD").is_dir():
            return parent
    raise FileNotFoundError(
        "未找到包含 AGENTS.md 和 研究方向/多径DPD 的仓库根目录。"
        "请把 ZIP 内的完整目录放入 DPD-Strong-Weak 仓库，直接运行本文件。"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main() -> None:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    from branch_probe import selftest, main as probe_main

    # Run the same Python interpreter selected in PyCharm, no hard-coded installation path.
    # Fixed default scope, deliberately not configurable by source edits or command line.
    test = HERE / "test_branch_probe.py"
    base = root / "研究方向" / "多径DPD"
    frozen = base / "实验" / "09_文献后机制核验" / "20261107" / "运行数据" / "common18"
    cases = frozen / "rows.json"
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = HERE.parent / "运行数据" / ("pilot3_"+now)
    out.mkdir(parents=True,exist_ok=False)
    status = {"experiment": "第20轮_分支可区分性核验",
              "mode": "ONE_CLICK_DEVELOPMENT_PILOT_ONLY",
              "cases": [0,8,16],"newly_created_output": str(out),
              "repo":str(root),"entry_script":str(Path(__file__)),
              "entry_sha256":sha(Path(__file__)),"probe_sha256":sha(HERE/'branch_probe.py'),
              "test_sha256":sha(test),"interpreter":sys.executable,
              "status":"started","results_not_claimed_without_real_iq":True}
    write_json(out/'运行状态.json',status)
    try:
        print("第20轮：先进行数学自检...",flush=True)
        status['selftest']=selftest()
        write_json(out/'运行状态.json',status)
        print("进行合成四站模型的程序单元检查...",flush=True)
        with (out/'合成单元检查.log').open('w',encoding='utf-8') as log:
            pr=subprocess.run([sys.executable,'-X','utf8',str(test)],cwd=str(root),
                              stdout=log,stderr=subprocess.STDOUT,check=False)
        if pr.returncode:
            raise RuntimeError('合成单元检查失败；详见 合成单元检查.log')
        status['model_control']='PASS'
        write_json(out/'运行状态.json',status)
        if not cases.is_file():
            status['status']='NEEDS_FROZEN_COMMON18_INPUT'
            status['note']='自检完成，但原冻结 common18/rows.json 不存在；未运行任何冻结 IQ 配对，也未生成新样本冒充原观测。'
            write_json(out/'运行状态.json',status)
            print('\n'+status['note'], '\n所需路径：', cases,flush=True)
            return
        print('进入3条冻结开发 IQ 短测（0、8、16）...',flush=True)
        # Explicit mode, without changing PyCharm Run Configuration.
        before=sys.argv[:]
        try:
            sys.argv=[str(HERE/'branch_probe.py'),'--pilot','--case-indices','0,8,16',
                      '--output',str(out/'冻结IQ诊断')]
            probe_main()
        finally:
            sys.argv=before
        status['status']='PILOT3_COMPLETED'
        status['blind_report']=str(out/'冻结IQ诊断'/'completion.json')
        write_json(out/'运行状态.json',status)
        print('\n3条冻结IQ短测完成；结果目录：',out,flush=True)
    except Exception as exc:
        status['status']='FAILED_OR_INCOMPLETE'
        status['error']=repr(exc)
        status['traceback']=traceback.format_exc()
        write_json(out/'运行状态.json',status)
        print('\n第20轮未完成，已保存错误与已有审计记录：',out,flush=True)
        raise

if __name__ == '__main__':
    main()
