"""Explicit project-local relocation; it never redirects to another project."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def registry():
    return json.loads((ROOT/'项目维护/布局迁移_20261106/映射.json').read_text(encoding='utf-8'))


def relocate(value, *, original=False):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT/path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        return path
    rel = path.relative_to(ROOT).as_posix()
    info = registry()
    for before, after in sorted(info['rules'].items(), key=lambda p: len(p[0]), reverse=True):
        if rel == before or rel.startswith(before+'/'):
            rel = after+rel[len(before):]
            break
    if original and rel in info.get('modified_originals', {}):
        rel = info['modified_originals'][rel]
    else:
        rel = info.get('code_files', {}).get(rel, rel)
    return ROOT/rel
