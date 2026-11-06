"""Explicit single-record execution; never starts or resumes a formal batch."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
from 研究方向.多径DPD.代码.mp_dpd import api  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='四站已保存 IQ 的单条定位；不重启历史批次')
    parser.add_argument('--input', required=True, type=Path, help='含 iq 字段的 NPZ')
    parser.add_argument('--method', choices=['MP', 'FLOOD', 'XIA'], default='MP')
    parser.add_argument('--seed', type=int, default=2026110426)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    inp = args.input.resolve(strict=True)
    if not inp.is_relative_to(ROOT):
        raise ValueError('Input must be inside the active project')
    out = args.output or api.OUTPUT/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.method+'.json')
    out = out.resolve()
    if not out.is_relative_to(ROOT/'研究方向/多径DPD/实验') or out.exists():
        raise ValueError('Use a new output file under 研究方向/多径DPD/实验')
    with np.load(inp, allow_pickle=False) as source:
        iq = source['iq'].copy()
    with threadpool_limits(limits=2):
        result = api.locate(iq, args.method, args.seed)
    api.runtime().cfg.write(out, dict(input=str(inp), method=args.method, result=result))
    print('结果：', out)


if __name__ == '__main__':
    main()
