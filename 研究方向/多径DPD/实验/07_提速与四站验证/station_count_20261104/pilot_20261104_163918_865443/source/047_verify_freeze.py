"""Read-only checksum validation before reusing MP_DPD_GATED_V1."""
import hashlib
import json
from pathlib import Path


def main():
    folder=Path(__file__).resolve().parent
    frozen=json.loads((folder/'冻结清单.json').read_text(encoding='utf-8'))
    failed=[]
    for name,digest in frozen['sources'].items():
        path=Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            failed.append(name)
    if failed:raise RuntimeError('Frozen source mismatch: '+repr(failed))
    print(f"{frozen['version']}: {len(frozen['sources'])} source hashes verified; no simulation started.")


if __name__=='__main__':main()
