"""Read-only historical tree inventory; writes only this audit's manifests."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('F:/DPD_Strong_Weak')
BASE = ROOT/'outputs_e2e/research_admission_20261027'
OUT = Path(__file__).resolve().parent
SNAPSHOT_NAMES = {'source', 'sources_snapshot', 'source_snapshot', 'source_snapshots', 'snapshot', 'snapshots'}


def source_copy(path):
    return any(p.lower() in SNAPSHOT_NAMES for p in path.relative_to(BASE).parts[:-1])


def main():
    files = sorted(p for p in BASE.rglob('*') if p.is_file())
    dirs = [p for p in BASE.rglob('*') if p.is_dir()]
    manifest = []
    refs = []
    for path in files:
        stat = path.stat()
        item = {'path': path.relative_to(ROOT).as_posix(), 'bytes': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'extension': path.suffix,
                'snapshot_path_heuristic': source_copy(path)}
        if path.suffix == '.py':
            raw = path.read_bytes()
            item['sha256'] = hashlib.sha256(raw).hexdigest()
            if not source_copy(path):
                for number, line in enumerate(raw.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    if re.search(r'sys\.path|importlib|research_admission_|parents\[|BASE\s*=|ROOT\s*=', line):
                        refs.append({'path': item['path'], 'line': number, 'text': line[:700]})
        manifest.append(item)
    summaries = []
    roots = [BASE, BASE/'xia_pairing', BASE/'localization_identifiability']
    roots += sorted(p for p in (BASE/'localization_identifiability').iterdir() if p.is_dir())
    for path in roots:
        subset = [p for p in files if p.is_relative_to(path)]
        py = [p for p in subset if p.suffix == '.py']
        summaries.append({'path': path.relative_to(ROOT).as_posix(), 'files': len(subset),
                          'directories_excluding_self': sum(p.is_relative_to(path) for p in dirs)-int(path != BASE),
                          'bytes': sum(p.stat().st_size for p in subset), 'python': len(py),
                          'python_snapshot_path_heuristic': sum(source_copy(p) for p in py),
                          'python_unique_content': len({hashlib.sha256(p.read_bytes()).hexdigest() for p in py})})
    external = []
    probes = list(ROOT.glob('*.md'))
    for folder in ['运行入口', '统一模型代码']:
        probes.extend((ROOT/folder).rglob('*.py'))
    for path in probes:
        for number, line in enumerate(path.read_text(encoding='utf-8-sig', errors='replace').splitlines(), 1):
            if 'research_admission_20261027' in line or 'localization_identifiability' in line:
                external.append({'path': path.relative_to(ROOT).as_posix(), 'line': number, 'text': line[:1200]})
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': str(BASE), 'mode': 'STATIC_READ_ONLY_NO_IMPORT_NO_EXPERIMENT',
              'manifest_note': 'Metadata for all files; SHA256 only Python files. Snapshot classification is path-name heuristic, not authority.',
              'summaries': summaries, 'extension_counts': dict(Counter(p.suffix for p in files)),
              'internal_reference_lines': len(refs), 'external_reference_lines': len(external)}
    for name, value in [('inventory_summary.json', record), ('file_manifest.json', manifest),
                        ('python_reference_lines.json', refs), ('external_reference_lines.json', external)]:
        (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'totals': summaries[:3], 'internal_reference_lines': len(refs),
                      'external_reference_lines': len(external)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
