"""Static contract and artifact audit only; does not execute a locator."""
from pathlib import Path
import ast
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
manifest = json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
checks = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
          for p, digest in manifest['identities'].items()}
tree = ast.parse((ROOT/'boundary_update.py').read_text(encoding='utf-8'))
called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
checks['backend_does_not_call_reduced_step'] = 'reduced_step' not in called
checks['backend_calls_position_jac'] = 'position_jac' in called
step = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'step_problem')
checks['step_declares_80_dimension'] = 'np.eye(80)' in ast.unparse(step)
run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
checks['original_60_iteration_limit'] = 'range(60)' in ast.unparse(run)
checks['power_dominance_counterexample'] = 2*abs(-.5)**2 < 1 and 1-.5-.5 == 0
report = OUT/'公式级贡献对照.md'
missing = []
for link in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
    if not link.startswith(('http://', 'https://', '#')) and not (report.parent/link).exists():
        missing.append(link)
checks['local_report_links'] = not missing
assert all(checks.values()), [k for k, value in checks.items() if not value]
(OUT/'audit.json').write_text(json.dumps(dict(checks=checks, missing_links=missing,
    report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    kind='STATIC_CODE_AND_ALGEBRA_CHECK_NOT_LOCALIZATION_PERFORMANCE'), ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS', len(checks), 'static/identity/algebra checks; no localization run')
