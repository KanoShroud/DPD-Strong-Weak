"""Resolve recorded historical paths without rewriting evidence identities."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
OLD = ROOT/'outputs_e2e/research_admission_20261027'
MANIFEST = ROOT/'outputs_e2e/mp_dpd/migration_20261106/migration.json'


@lru_cache(maxsize=1)
def _mapping():
    doc = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if doc['status'] not in ('copied_verified', 'moved_verified'):
        raise RuntimeError('Historical migration is not verified')
    files, directories, ambiguous = {}, {}, set()
    for row in doc['files']:
        old, new = ROOT/row['old'], ROOT/row['new']
        # Hash-bearing historical records refer to original prose bytes; the
        # reader-facing relocated reports have only their links repaired.
        files[old] = ROOT/doc.get('navigation_originals', {}).get(row['new'], row['new'])
        a, b = old.parent, new.parent
        while a.is_relative_to(OLD) and a != OLD:
            if a in directories and directories[a] != b:
                ambiguous.add(a)
            directories[a] = b
            a, b = a.parent, b.parent
    for a in ambiguous:
        directories.pop(a, None)
    return files, sorted(directories.items(), key=lambda pair: len(pair[0].parts), reverse=True)


def resolve_history(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT/path
    path = path.resolve()
    if not path.is_relative_to(OLD):
        return path
    files, directories = _mapping()
    if path in files:
        return files[path]
    for before, after in directories:
        if path.is_relative_to(before):
            return after/path.relative_to(before)
    raise FileNotFoundError(f'No unique migrated path (possibly a removed cache): {path}')
