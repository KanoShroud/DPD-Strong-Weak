"""G9独立输出、冻结源码合同、资源与累计预算。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs_e2e/unified/e2e_g9'
FROZEN = Path('F:/SourceCount_DPD/outputs').resolve()


def location(path):
    path = Path(path).resolve()
    if path == BASE.resolve() or not path.is_relative_to(BASE.resolve()) or path.is_relative_to(FROZEN):
        raise ValueError('G9输出根越界')
    return path


def write(path, value):
    path = location(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source_manifest():
    paths = list((ROOT/'统一模型代码/gates/g9').glob('*.py'))
    paths += [ROOT/'统一模型代码/gates/g8'/name for name in ('scenes.py', 'physics.py')]
    paths += list((ROOT/'运行入口/E2E/G9').glob('*.py'))
    return [identity(p) for p in sorted(paths)]


def verify(manifest):
    for item in manifest:
        if identity(item['path']) != item:
            raise RuntimeError('冻结源码/输入身份变化：'+item['path'])


class Guard:
    def __init__(self, seconds, consumed=0.):
        self.start = time.perf_counter()
        self.limit, self.consumed = seconds, consumed
        self.last = 0.

    def __call__(self):
        now = time.perf_counter()
        if now-self.start+self.consumed >= self.limit:
            raise RuntimeError('G9累计预算达到上限；保留结果，禁止自动放宽')
        if now-self.last < 1:
            return
        self.last = now
        if psutil.virtual_memory().percent >= 85:
            raise RuntimeError('RAM达到85%')
        if shutil.disk_usage(ROOT).free < 50*2**30:
            raise RuntimeError('磁盘剩余不足50GiB')
        if torch.cuda.memory_allocated() >= 14*2**30:
            raise RuntimeError('CUDA占用达到14GiB')


def preflight(out):
    out = location(out)
    if out.exists():
        raise FileExistsError('新执行必须使用新目录：'+str(out))
    if not torch.cuda.is_available():
        raise RuntimeError('G9需要现有CUDA环境')
    Guard(1)()
    out.mkdir(parents=True)
    return out
