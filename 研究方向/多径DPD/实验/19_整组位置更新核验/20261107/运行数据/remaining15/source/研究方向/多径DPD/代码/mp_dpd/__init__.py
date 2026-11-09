"""Four-station blind multipath DPD; no result-directory source imports."""
from pathlib import Path


def code_path(relative):
    base = Path(__file__).resolve().parent
    result = (base/relative).resolve()
    if not result.is_relative_to(base) or result.suffix != '.py':
        raise ValueError('Expected an internal Python module path')
    return result
