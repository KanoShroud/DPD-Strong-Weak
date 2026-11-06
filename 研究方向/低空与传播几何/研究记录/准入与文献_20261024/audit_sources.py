"""Merge read-only PDF identities and run structural preflight (not semantic QA)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREFLIGHT = Path('C:/Users/Administrator/.codex/skills/academic-research-suite/ars/scripts/pdf_read_preflight.py')


def main():
    combined = {}
    for filename in ['sources.json', 'admission_source_identity.json',
                     'link_sources/sources.json', 'backup_sources/sources.json']:
        for entry in json.loads((ROOT / filename).read_text(encoding='utf-8')):
            path = Path(entry.get('path', entry.get('source'))).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'PDF identity drift: {path}')
            row = combined.setdefault(digest, {'path': str(path), 'sha256': digest, 'registries': []})
            row['registries'].append(filename)
    folder = ROOT / 'structural_checks'
    folder.mkdir(exist_ok=True)
    for digest, row in combined.items():
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', str(PREFLIGHT), row['path']],
            capture_output=True, text=True, encoding='utf-8', errors='replace', check=True,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8:replace'},
        )
        integrity = json.loads(result.stdout)
        row['structural_verdict'] = integrity['verdict']
        row['pages'] = integrity['reader_page_count']
        (folder / f'{digest[:12]}.json').write_text(
            json.dumps(integrity, ensure_ascii=False, indent=2), encoding='utf-8')
    output = {'unique_local_pdfs': len(combined), 'records': list(combined.values()),
              'scope': 'Structural integrity and frozen identity only; reading scope is in the reports.'}
    (ROOT / 'source_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'unique_local_pdfs': len(combined),
                      'verdicts': sorted({r['structural_verdict'] for r in combined.values()})}))


if __name__ == '__main__':
    main()
