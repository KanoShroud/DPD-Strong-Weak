"""Result-only regression for the layout warning and completed-run shortcut."""
import importlib.util
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import warnings

from . import report, solvers, runner, runtime
from .config import BASE, ROOT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The frozen source snapshot is a read-only reference, including caches.
    with patch.object(sys, 'dont_write_bytecode', True):
        spec.loader.exec_module(module)
    return module


def main():
    launcher = load_module('g10r2_launcher_check', ROOT/'运行入口/E2E/G10/G10R2双候选比较一键运行.py')
    source = launcher.completed_run(BASE)
    if source is None:
        raise RuntimeError('This regression requires a completed development run')
    before_hashes = {str(p.relative_to(source)): runtime.sha(p) for p in source.rglob('*') if p.is_file()}
    target = runtime.fresh('plot_layout_check')
    if target.exists() or not target.resolve().is_relative_to(BASE.resolve()):
        raise RuntimeError('Regression requires a new isolated output')
    target.mkdir()
    runtime.write(target/'original_artifacts_before.json', before_hashes)
    old = load_module('统一模型代码.gates.g10.r2._frozen_report_check',
                      source/'source_snapshot/统一模型代码/gates/g10/r2/report.py')
    results = {}
    with patch.object(solvers, 'run', side_effect=AssertionError('No localization allowed')), \
         patch.object(runner, 'develop', side_effect=AssertionError('No development allowed')):
        for name, module in (('before', old), ('after', report)):
            out = target/name
            out.mkdir()
            shutil.copyfile(source/'contract.json', out/'contract.json')
            shutil.copytree(source/'rows', out/'rows')
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                module.generate(out)
            results[name] = [str(w.message) for w in captured]
        launcher.main()

    assert results['before'].count('The figure layout has changed to tight') == 6, results['before']
    assert not results['after'], results['after']
    # Identical CSV/statistics and PNG bytes establish that only the warning
    # changed; PDF metadata and plot-code provenance legitimately differ.
    compared = []
    for path in (target/'before').rglob('*'):
        if path.is_file() and (path.suffix in ('.csv', '.png', '.md', '.tex') or
                              path.name in ('comparison_report.json', 'mechanism_report.json')):
            rel = path.relative_to(target/'before')
            assert runtime.sha(path) == runtime.sha(target/'after'/rel), str(rel)
            compared.append(str(rel))
    bad_name = next(iter(runtime.read(source/'final_audit_report.json')['rows']))
    original_sha = runtime.sha

    def bad_sha(path):
        return 'invalid' if Path(path) == source/'rows'/bad_name else original_sha(path)

    with patch.object(runtime, 'sha', side_effect=bad_sha):
        try:
            launcher.completed_run(BASE)
        except RuntimeError as exc:
            assert '结果身份变化' in str(exc), str(exc)
        else:
            raise AssertionError('Changed result was not rejected')
    after_hashes = {str(p.relative_to(source)): runtime.sha(p) for p in source.rglob('*') if p.is_file()}
    assert before_hashes == after_hashes, 'Original evidence changed'
    runtime.write(target/'validation_report.json', dict(
        passed=True, source=str(source), source_artifacts_unchanged=True,
        warning_messages=results, exact_matching_outputs=compared,
        solver_called=False, changed_result_rejected=True, completed_entry_checked=True,
        original_artifacts=before_hashes,
        repaired_sources={str(p.relative_to(ROOT)): runtime.sha(p) for p in (
            ROOT/'统一模型代码/gates/g10/r2/report.py',
            ROOT/'运行入口/E2E/G10/G10R2双候选比较一键运行.py', Path(__file__))}))
    print('PASS: no warnings, identical statistics/PNG, original evidence unchanged; '+str(target), flush=True)


if __name__ == '__main__':
    main()
