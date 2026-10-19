"""Protect prior formulas/tables and check saved-only closeout statements."""
import hashlib
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]


def main():
    text = (PAPER / 'paper_draft.tex').read_text(encoding='utf-8')
    old = (PAPER / 'archive/contribution_closeout_20261019/paper_draft.tex').read_text(encoding='utf-8')
    equations = re.findall(r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}', old, re.S)
    assert all(e in text for e in equations)
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', old, re.S)
    assert all(t in text for t in tables)
    a = json.loads((PAPER / 'data/contribution_closeout_20261019/analysis.json').read_text(encoding='utf-8'))
    for path, digest in a['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert a['ap5_recovered']['observations'] == 96
    assert a['ap5_recovered']['layout_repeat_count'] == 18
    assert a['ap5_failed']['observations'] == 24
    assert a['ap5_failed']['layout_repeat_count'] == 6
    for method, count in [('MUSIC', 1), ('MVDR', 0)]:
        for variant in ('rect4', 'rect8', 'hann4', 'hann8', 'hann_long4'):
            assert a['spectral_counts'][f'{method}/{variant}/actual/15.0'] == dict(n=4, success=count)
    assert '其对定位结果的具体贡献尚未分离' not in text
    assert r'\label{app:stoppingprecision}' in text
    result = dict(passed=True, preserved_equations=len(equations), preserved_tables=len(tables),
                  saved_evidence_identity=True, recovery_unit_checked=True,
                  spectral_diagnostic_checked=True, no_new_positioning=True)
    out = PAPER / 'qa/contribution_closeout_20261019'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'content_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
