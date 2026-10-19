"""Explicit relocation of recorded input paths; never rewrite stored contracts."""
from pathlib import Path
import os
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = Path('F:/SourceCount_DPD')
OLD_ROOTS = (Path('F:/SourceCount_DPD_E2E'), Path('F:/DPD_WeakSource_Localization'))
PROJECTS = tuple(Path('F:/') / n for n in
                 ('DPD_Strong_Weak', 'DPD_EndToEnd_Learning', 'DPD_Thesis_Reproduction'))
SOURCE_ALIASES_USED = set()


def input_path(value, *, strict=False):
    """Resolve local history or the explicitly allowed frozen original project."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    for old in OLD_ROOTS:
        if path.is_relative_to(old):
            path = ROOT / path.relative_to(old)
            break
    # Legacy chapter outputs are INPUTS only. New outputs use outputs_e2e.
    if path.is_relative_to(ROOT / 'outputs'):
        path = REFERENCE_ROOT / 'outputs' / path.relative_to(ROOT / 'outputs')
    path = path.resolve(strict=strict)
    if any(p != ROOT and path.is_relative_to(p) for p in PROJECTS):
        raise ValueError(f'Cross-project input is not allowed: {path}')
    return path


def local_path(value):
    path = input_path(value, strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Expected a local evidence snapshot: {path}')
    return path


def _source_alias(row):
    if not isinstance(row, dict) or not {'path', 'sha256', 'size_bytes'} <= row.keys():
        return None
    path = input_path(row['path'])
    if path.suffix != '.py' or not path.is_relative_to(ROOT):
        return None
    registry = ROOT / 'docs/path_migration_20261019/source_aliases.json'
    if not registry.exists():
        return None
    records = json.loads(registry.read_text(encoding='utf-8'))
    rel = path.relative_to(ROOT).as_posix()
    alias = records.get(rel)
    if not alias or row['sha256'] not in (alias['before_sha256'], alias['after_sha256']):
        return None
    expected_size = alias['before_bytes'] if row['sha256'] == alias['before_sha256'] else alias['after_bytes']
    if row['size_bytes'] != expected_size:
        raise ValueError(f'Source size differs from registered migration version: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != alias['after_sha256']:
        raise ValueError(f'Source changed after path migration registration: {path}')
    original = ROOT / alias['original']
    payload = original.read_bytes()
    if hashlib.sha256(payload).hexdigest() != alias['before_sha256']:
        raise ValueError(f'Original source archive identity changed: {original}')
    SOURCE_ALIASES_USED.add(rel)
    return alias


def verified_source_path(row):
    """Old source hashes refer to archived bytes plus a checked current-code alias."""
    alias = _source_alias(row)
    if alias and row['sha256'] == alias['before_sha256']:
        return ROOT / alias['original']
    return input_path(row['path'])


def identities_equal(actual, expected):
    """Canonical paths and explicitly registered path-only source versions only."""
    def canonical(value):
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, dict):
            result = {k: canonical(v) for k, v in value.items()}
            if isinstance(value.get('path'), str):
                result['path'] = str(input_path(value['path']))
                alias = _source_alias(value)
                if alias:
                    result['sha256'] = alias['before_sha256']
                    result['size_bytes'] = alias['before_bytes']
            return result
        return value
    return canonical(actual) == canonical(expected)


def install_write_guard():
    """Reject Python filesystem writes to frozen or sibling project trees."""
    protected = (REFERENCE_ROOT, *OLD_ROOTS, *(p for p in PROJECTS if p != ROOT))
    def audit(event, args):
        paths = []
        if event == 'open' and len(args) > 2:
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                paths = args[:1]
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            paths = args[:1]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            paths = args[:2]
        for value in paths:
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if any(path.is_relative_to(p) for p in protected):
                    raise PermissionError(f'Read-only reference or other project: {path}')
    sys.addaudithook(audit)
