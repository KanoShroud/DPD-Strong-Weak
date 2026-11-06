"""Finalize scoped migration verification without per-file hashes."""
import ast
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/'outputs_e2e/mp_dpd/migration_20261106'
MANIFEST = RECEIPT/'migration.json'
HISTORY = ROOT/'outputs_e2e/mp_dpd/history'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        self.values.extend(value for key, value in attrs if key in ('href', 'src') and value)


def main():
    doc = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert doc['status'] == 'moved_verified'
    assert not (ROOT/'outputs_e2e/research_admission_20261027').exists()
    copies = doc.get('navigation_originals', {})
    html_files = list(HISTORY.rglob('*.html'))+list((ROOT/'outputs_e2e/literature_reaudit_20261105').glob('*.html'))
    for file in html_files:
        raw = file.read_bytes()
        text = raw.decode('utf-8-sig')
        fixed = re.sub(r'''((?:href|src)=["'])([A-Za-z]:[/\\])''', r'\1file:///\2', text)
        if text != fixed:
            if file.is_relative_to(HISTORY):
                key = file.relative_to(ROOT).as_posix()
                if key not in copies:
                    backup = RECEIPT/'navigation_originals'/file.relative_to(HISTORY)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(raw)
                    copies[key] = backup.relative_to(ROOT).as_posix()
            tmp = file.with_name(file.name+'.migration_pending')
            tmp.write_bytes((b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'')+fixed.encode('utf-8'))
            os.replace(tmp, file)
    doc['navigation_originals'] = copies
    doc['archived_entries'] = [dict(original=p.relative_to(ROOT).as_posix(),
        entry=(Path('运行入口/E2E/G11')/p.name).as_posix())
        for p in (HISTORY/'00_早期公共脚本与协议/旧批量入口').glob('*.py')]
    assert len(doc['archived_entries']) == 7
    for row in doc['files']:
        target = ROOT/row['new']
        assert target.is_file(), target
        original = ROOT/copies.get(row['new'], row['new'])
        assert original.stat().st_size == row['bytes'], original
    historical = {row['old']: ROOT/row['new'] for row in doc['files']}
    for row in doc['runtime']:
        original = ast.parse(historical[row['old']].read_text(encoding='utf-8-sig'))
        current = ast.parse((ROOT/row['new']).read_text(encoding='utf-8'))

        def normalize(tree):
            tree.body = [node for node in tree.body
                if not (isinstance(node, ast.FunctionDef) and node.name == 'main')
                and not (isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))
                and not (row['new'].endswith('/scene.py') and isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets))]
            return ast.dump(tree, include_attributes=False)

        assert normalize(original) == normalize(current), row['new']
    a, b = np.load(RECEIPT/'before.npz'), np.load(RECEIPT/'final.npz')
    assert a.files == b.files
    for key in a.files:
        assert np.array_equal(a[key], b[key], equal_nan=True), key
    loaded = json.loads((RECEIPT/'final.json').read_text(encoding='utf-8'))
    assert all('outputs_e2e' not in p for p in loaded['sources'])
    missing, total = [], 0
    for file in html_files:
        parser = Links()
        parser.feed(file.read_text(encoding='utf-8-sig'))
        for value in parser.values:
            if value.startswith(('#', 'http:', 'https:', 'mailto:', 'data:', 'javascript:')):
                continue
            parsed = urlsplit(value)
            rawpath = unquote(parsed.path)
            if parsed.scheme == 'file' and re.match(r'^/[A-Za-z]:', rawpath):
                rawpath = rawpath[1:]
            path = Path(rawpath)
            if not path.is_absolute():
                path = file.parent/path
            total += 1
            if not path.exists():
                missing.append(dict(page=file.relative_to(ROOT).as_posix(), href=value, target=str(path.resolve())))
    doc['verification'] = dict(files_preserved=len(doc['files']), equal_arrays=len(a.files),
        runtime_ast_equivalent=len(doc['runtime']),
        source_files_loaded=len(loaded['sources'])-1, old_source_tree_absent=True,
        html_pages=len(html_files), local_links=total, missing_links=missing,
        cleanup_bytes=sum(r['bytes'] for r in doc['cleanup']),
        notes='Fixed-input engineering equivalence only; existing estimator failure states preserved.')
    doc['temporary_cleanup'] = [name for name in [
        'tools/organize_mp_dpd.py',
        'outputs_e2e/mp_dpd/migration_20261106/after.npz',
        'outputs_e2e/mp_dpd/migration_20261106/after.json'] if not (ROOT/name).exists()]
    MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(doc['verification'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
