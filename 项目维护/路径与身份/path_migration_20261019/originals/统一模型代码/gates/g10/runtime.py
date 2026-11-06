"""Independent output root, byte-identical input consumption, resumable elapsed budget."""
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from .config import ROOT, BASE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()) or path == BASE.resolve():
        raise ValueError('G10 output outside isolated root')
    if path.is_relative_to(Path('F:/SourceCount_DPD/outputs').resolve()):
        raise ValueError('Frozen reference root')
    return path


def atomic_replace(source, destination):
    """Keep atomicity while tolerating brief Windows sharing/access conflicts."""
    delays = (.05, .1, .2, .4, .8, 1., 1.)
    for attempt in range(len(delays)+1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def write(path, data):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.pending')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest():
    files = list((ROOT/'统一模型代码/gates/g10').glob('*.py'))
    files += list((ROOT/'统一模型代码/gates/g9').rglob('*.py'))
    files += list((ROOT/'统一模型代码/gates/g8').glob('*.py'))
    files += list((ROOT/'运行入口/E2E/G10').glob('*.py'))
    files += list((ROOT/'统一模型代码/gates/g10').glob('*.md'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def verify(identity):
    for relative, expected in identity.items():
        path = (ROOT/relative).resolve(strict=True)
        if sha(path) != expected:
            raise RuntimeError('Frozen source changed: '+relative)


def resource_check():
    if psutil.virtual_memory().percent >= 85:
        raise RuntimeError('RAM >=85%; preserve progress')
    if shutil.disk_usage(ROOT).free < 50*2**30:
        raise RuntimeError('Disk free <50 GiB')
    if torch.cuda.memory_allocated() >= 14*2**30:
        raise RuntimeError('CUDA allocation >=14 GiB')


def preflight(out, contract=None):
    out = safe(out)
    if out.exists():
        raise FileExistsError('New run requires a fresh directory: '+str(out))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    resource_check()
    identity = manifest()
    verify(identity)
    out.mkdir(parents=True)
    write(out/'contract.json', dict(contract or {}, sources=identity,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                         gpu=torch.cuda.get_device_name(0)), created=time.time()))
    return out


def save_scene(out, key, scene):
    path = safe(out/'inputs'/f'{key}.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.with_suffix('.json').exists():
            raise FileExistsError(path)
        # Recovery after atomic data commit but before metadata commit. Never
        # replace completed data: accept only exactly regenerated array contents.
        with np.load(path, allow_pickle=False) as z:
            for field in ('signal', 'noise', 'positions'):
                if not np.array_equal(z[field], scene[field]):
                    raise RuntimeError('Uncommitted scene differs: '+str(path))
    else:
        pending = path.with_name(path.name+'.pending')
        with pending.open('wb') as stream:
            np.savez_compressed(stream, signal=scene['signal'], noise=scene['noise'], positions=scene['positions'])
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(pending, path)
    info = dict(scene['meta'], path=str(path.relative_to(out)), sha256=sha(path))
    write(path.with_suffix('.json'), info)
    return info


def load_scene(out, key):
    info = read(out/'inputs'/f'{key}.json')
    payload = (out/info['path']).resolve(strict=True).read_bytes()
    if hashlib.sha256(payload).hexdigest() != info['sha256']:
        raise RuntimeError('Input identity mismatch: '+key)
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        return dict(signal=z['signal'], noise=z['noise'], positions=z['positions'], meta=info)


class Clock:
    def __init__(self, out, limit):
        self.path = out/'ledger.json'
        self.state = read(self.path) if self.path.exists() else dict(consumed=0., pending=None, interruptions=[])
        if self.state['pending']:
            # A killed process cannot report its last wall interval: charge reserved
            # per-task budget, record it, then allow a fresh task attempt.
            self.state['consumed'] += self.state['pending']['reserve']
            self.state['interruptions'].append(self.state['pending'])
            self.state['pending'] = None
        self.limit, self.mark, self.checked = limit, time.perf_counter(), 0.
        write(self.path, self.state)

    def consumed(self):
        return self.state['consumed']+time.perf_counter()-self.mark

    def check(self):
        if self.consumed() >= self.limit:
            raise RuntimeError('G10 cumulative time budget reached; no automatic extension')
        if time.perf_counter()-self.checked >= 1:
            resource_check()
            self.checked = time.perf_counter()

    def checkpoint(self, pending=None):
        self.state['consumed'] = self.consumed()
        self.state['pending'] = pending
        self.mark = time.perf_counter()
        write(self.path, self.state)


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.start, self.last = total, label, time.perf_counter(), 0.

    def update(self, done, detail='', force=False):
        now = time.perf_counter()
        if not force and now-self.last < 30 and done != self.total:
            return
        elapsed = now-self.start
        eta = elapsed/max(done, 1)*(self.total-done)
        n = int(24*done/max(self.total, 1))
        bar = '#'*n+'-'*(24-n)
        text = f'{self.label} [{bar}] {done}/{self.total} | {elapsed/60:.1f} min | ETA {eta/60:.1f} min {detail}'
        print(text, end='\r' if sys.stdout.isatty() and done < self.total else '\n', flush=True)
        self.last = now
