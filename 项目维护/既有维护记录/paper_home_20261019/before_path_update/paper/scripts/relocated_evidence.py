"""Map recorded old paths to local copied evidence without editing records."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = Path('F:/SourceCount_DPD_E2E')


def local_path(value):
    path = Path(value)
    if path.is_absolute():
        try:
            path = ROOT / path.relative_to(OLD_ROOT)
        except ValueError:
            pass
    else:
        path = ROOT / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'External evidence is not authorized by this resolver: {path}')
    return path
