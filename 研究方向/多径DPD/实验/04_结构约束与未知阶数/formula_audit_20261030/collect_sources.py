"""Archive public bibliographic responses and local code identities, no simulation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def main():
    log = []
    urls = {
        'Friedlander2014_crossref': 'https://api.crossref.org/works/10.1109%2FACSSC.2014.7094643',
        'Du2018_crossref': 'https://api.crossref.org/works/10.3390%2Fs18030892',
        'Du2019_crossref': 'https://api.crossref.org/works/10.1109%2FACCESS.2018.2889958',
        'Weiss2022_crossref': 'https://api.crossref.org/works/10.1109%2FTSP.2022.3173731',
        'VP2013_crossref': 'https://api.crossref.org/works/10.1007%2Fs10589-012-9492-9',
        'Friedlander2014_program': 'https://asilomarsscconf.org/webpage/asil14/new/AS14_BoA_Content_v004_WithCover.pdf',
    }
    for name, url in urls.items():
        record = dict(name=name, url=url)
        try:
            response = requests.get(url, timeout=25)
            record.update(status=response.status_code, final_url=response.url)
            response.raise_for_status()
            suffix = '.pdf' if name.endswith('program') else '.json'
            path = OUT/(name+suffix)
            if suffix == '.json':
                body = response.json()
                path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
            else:
                assert response.content.startswith(b'%PDF')
                path.write_bytes(response.content)
            record.update(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as exc:
            record['error'] = str(exc)
        log.append(record)
        print(name, record.get('status'), record.get('error', 'saved'), flush=True)
    source_paths = [ROOT/name for name in ['blind_backend_pilot.py', 'boundary_update.py',
        'solver_reliability.py', 'position_reduction.py', 'blind_initialization.py', 'structural_probe.py',
        'flood_band_pair.py', 'bounded_consensus.py']]
    source_paths += [ROOT/'conditional_20261029_112305_738322/literature/FH72DMFX.txt',
                    OUT/'Du2019_fulltext.txt', OUT/'Weiss2022_fulltext.txt', Path(__file__)]
    identities = {str(p.resolve(strict=True)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (OUT/'source_manifest.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        requests=log, identities=identities, kind='LITERATURE_AND_STATIC_CODE_AUDIT_NO_NEW_SIMULATION'),
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
