"""R1 source/data identity, atomic results, safe resume; no wall-clock solver cap."""
from datetime import datetime
import shutil
import time
from ..runtime import read, write, sha, verify, resource_check, save_scene, load_scene
from ..runtime import preflight as original_preflight
from .config import ROOT, BASE

__all__ = ['read', 'write', 'sha', 'verify', 'resource_check', 'save_scene', 'load_scene',
           'sources', 'preflight', 'fresh', 'Ledger']


def fresh(stage):
    return BASE/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))


def sources():
    paths = list((ROOT/'统一模型代码/gates/g10/r1').glob('*.py'))
    paths += list((ROOT/'统一模型代码/gates/g10/r1').glob('*.md'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def preflight(out, config):
    original_preflight(out, dict(config, r1_sources=sources()))
    contract = read(out/'contract.json')
    for relative, expected in {**contract['sources'], **contract['r1_sources']}.items():
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
        if sha(destination) != expected:
            raise RuntimeError('Source changed during snapshot: '+relative)


class Ledger:
    def __init__(self, out):
        self.path = out/'ledger.json'
        self.state = read(self.path) if self.path.exists() else dict(seconds=0., pending=None, interruptions=[])
        if self.state['pending']:
            pending = self.state['pending']
            pending['recorded_attempt_seconds'] = max(0., self.state['seconds']-pending.get('ledger_start_seconds', self.state['seconds']))
            pending['unrecorded_tail_note'] = 'Abrupt kills may lose at most one 30s ledger interval; not assumed zero.'
            self.state['interruptions'].append(pending)
            self.state['pending'] = None
        self.mark = time.perf_counter()
        self.last_flush = self.mark
        self.flush()

    def flush(self, pending=None, keep=False):
        now = time.perf_counter()
        self.state['seconds'] += now-self.mark
        self.mark = now
        if not keep:
            self.state['pending'] = pending
        write(self.path, self.state)
        self.last_flush = now

    def check(self):
        resource_check()
        if time.perf_counter()-self.last_flush >= 30:
            self.flush(keep=True)

    def total(self):
        return self.state['seconds']+time.perf_counter()-self.mark


def finish_audit(out):
    contract = read(out/'contract.json')
    verify(contract['sources'])
    verify(contract['r1_sources'])
    inputs = {}
    for p in (out/'inputs').glob('*.json'):
        info = read(p)
        actual = sha(out/info['path'])
        if actual != info['sha256']:
            raise RuntimeError('Input changed: '+str(p))
        inputs[p.name] = actual
    state_files = {}
    for p in out.rglob('manifest.json'):
        info = read(p)
        if info.get('version') != 'SBL_SHARED_STATE_V1':
            continue
        for entry in info['states'].values():
            state = (p.parent/entry['path']).resolve(strict=True)
            if not state.is_relative_to(out.resolve()) or sha(state) != entry['sha256']:
                raise RuntimeError('SBL state identity mismatch')
            state_files[str(state.relative_to(out))] = entry['sha256']
        state_files[str(p.relative_to(out))] = sha(p)
    result = dict(identity_passed=True, inputs=inputs, states=state_files,
                  output_replays={str(p.relative_to(out)):sha(p) for p in (out/'sbl_outputs').glob('*.json')},
                  rows={p.name: sha(p) for p in (out/'rows').glob('*.json')})
    write(out/'final_audit_report.json', result)
    return result
