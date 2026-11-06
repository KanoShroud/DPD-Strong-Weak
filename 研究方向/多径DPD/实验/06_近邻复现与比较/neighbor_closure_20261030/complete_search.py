"""Close nearest-neighbor retrieval branches and preserve source metadata."""
import hashlib
import json
import time

import fitz
import requests

from collect import OUT, fetch

DOIS = ['10.1109/JSAIT.2023.3287823', '10.1016/j.sigpro.2024.109484',
        '10.1109/LSP.2023.3328273', '10.1109/29.1552', '10.1109/78.324732',
        '10.1016/j.sigpro.2020.107907', '10.1016/j.dsp.2025.105290',
        '10.1016/j.sigpro.2026.110527', '10.1109/ACSSC.2016.7869551',
        '10.1016/j.sigpro.2014.02.010']
PDFS = {
    'Feder1988': 'https://www.eng.tau.ac.il/~meir/articles/3%20Parameter%20Estimation%20of%20Superimposed%20Signals%20Using%20the%20EM.pdf',
    'SAGE1994': 'https://web.eecs.umich.edu/~fessler/papers/files/jour/94/pre/tsp%2Csage.pdf',
    'DBD2023_author': 'https://arxiv.org/pdf/2208.04381',
    'DBD2024_author': 'https://arxiv.org/pdf/2303.13609',
    'Fascista2021_author': 'https://arxiv.org/pdf/1808.00857',
    'Papakonstantinou2008': 'https://www.eurecom.edu/publication/2676/download/cm-publi-2676.pdf',
}


def main():
    results = {}
    for i, doi in enumerate(DOIS):
        results[doi] = fetch('https://api.crossref.org/works/' + doi)
        print('metadata', i, results[doi].get('status'), flush=True)
        time.sleep(.6)
    for title in [
        'Robust LOS Identification for Passive Multi-Target Localization in Multipath Obstructed Environments',
        'Direct Position Determination with Distributed Sensor Arrays Under Multi-path Environment Matrix Reconstruction and Subspace Data Fusion',
    ]:
        results[title] = fetch('https://api.crossref.org/works', {'query.title': title, 'rows': 3})
    (OUT / 'nearest_metadata.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    files = []
    for name, url in PDFS.items():
        try:
            r = requests.get(url, timeout=30)
            rec = {'name': name, 'url': url, 'status': r.status_code}
            if r.ok and r.content.startswith(b'%PDF'):
                path = OUT / (name + '.pdf')
                path.write_bytes(r.content)
                rec['sha256'] = hashlib.sha256(r.content).hexdigest()
                with fitz.open(path) as pdf:
                    rec['pages'] = len(pdf)
                    (OUT / (name + '.txt')).write_text('\n\n'.join(
                        f'PDF PAGE {i+1}\n' + p.get_text() for i,p in enumerate(pdf)), encoding='utf-8')
            files.append(rec)
            print('pdf', name, rec.get('pages', r.status_code), flush=True)
        except requests.RequestException as e:
            files.append({'name': name, 'url': url, 'error': str(e)})
    (OUT / 'public_fulltexts.json').write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding='utf-8')
    # Additional source-linked forward snowballing for new closest families.
    snow = {}
    for doi in ['10.1109/ACCESS.2018.2889958', '10.1109/LCOMM.2023.3347620',
                '10.1109/JSAIT.2023.3287823']:
        rec = fetch('https://api.openalex.org/works/https://doi.org/' + doi)
        wid = rec.get('data', {}).get('id', '').split('/')[-1]
        if wid:
            snow[doi] = fetch('https://api.openalex.org/works', {'filter': 'cites:' + wid, 'per-page': 200})
    (OUT / 'snowball_extended.json').write_text(json.dumps(snow, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE')


if __name__ == '__main__':
    main()
