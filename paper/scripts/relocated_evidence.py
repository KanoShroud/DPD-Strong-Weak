"""Map recorded old paths to local copied evidence without editing records."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_ROOTS = (Path('F:/SourceCount_DPD_E2E'), Path('F:/DPD_WeakSource_Localization'))


def local_path(value):
    path = Path(value)
    if path.is_absolute():
        for old_root in OLD_ROOTS:
            try:
                path = ROOT / path.relative_to(old_root)
                break
            except ValueError:
                continue
    else:
        path = ROOT / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'External evidence is not authorized by this resolver: {path}')
    return path
