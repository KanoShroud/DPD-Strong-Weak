"""Read-only literature retrieval; no localization runs or Zotero writes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path

import fitz
import requests

OUT = Path(__file__).resolve().parent
QUERIES = [
    'direct position determination unknown signals multipath',
    'direct localization multiple sources multipath unknown waveform',
    'geolocation multipath maximum likelihood Friedlander',
    'blind source localization convolutive mixtures multipath',
    'joint channel estimation localization unknown signals multipath',
    'direct position estimation multipath expectation maximization',
    'cochannel interference multipath direct localization',
    'multiple source localization multipath semiblind',
    'direct localization multipath sparse Bayesian',
    'direct localization multipath cross spectrum PARAFAC',
    'Maximum Likelihood Position Estimation in the presence of multipath',
    'direct localization detection multiple sources multi-path environments',
]
SEEDS = ['10.1109/ACSSC.2014.7094643', '10.1109/JSYST.2020.3005918',
         '10.3390/s18030892', '10.1109/TSP.2022.3173731',
         '10.1109/JIOT.2022.3228810', '10.1109/TWC.2025.3601543']


def fetch(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=25)
        obj = {'url': r.url, 'status': r.status_code}
        if r.ok:
            obj['data'] = r.json()
        return obj
    except (requests.RequestException, ValueError) as exc:
        return {'url': url, 'error': str(exc)}


def main():
    tasks = [('crossref_' + str(i), 'https://api.crossref.org/works',
              {'query.bibliographic': q, 'rows': 40}) for i, q in enumerate(QUERIES)]
    tasks += [('openalex_' + str(i), 'https://api.openalex.org/works/https://doi.org/' + doi,
               None) for i, doi in enumerate(SEEDS)]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(fetch, url, params): name for name, url, params in tasks}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            results[name] = fut.result()
            print(name, results[name].get('status', 'ERROR'), flush=True)
    for i, _ in enumerate(SEEDS):
        record = results.get('openalex_' + str(i), {}).get('data', {})
        if record.get('id'):
            results['citing_' + str(i)] = fetch('https://api.openalex.org/works',
                {'filter': 'cites:' + record['id'].split('/')[-1], 'per-page': 200})
    # Public read-only Zotero local API, all top-level metadata for DOI matching.
    inventory = []
    for start in range(0, 5000, 100):
        batch = fetch('http://127.0.0.1:23119/api/users/0/items/top',
                      {'format': 'json', 'limit': 100, 'start': start})
        data = batch.get('data', [])
        if not isinstance(data, list):
            results['zotero_error'] = batch
            break
        inventory.extend(data)
        if len(data) < 100:
            break
    (OUT / 'zotero_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'search_raw.json').write_text(json.dumps({'queries': QUERIES, 'seeds': SEEDS,
        'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    papers = {}
    for name, result in results.items():
        data = result.get('data', {})
        items = data.get('message', {}).get('items', []) if name.startswith('crossref_') else data.get('results', [])
        for it in items:
            doi = (it.get('DOI') or it.get('doi') or '').lower().replace('https://doi.org/', '')
            title = it.get('title', '')
            title = title[0] if isinstance(title, list) and title else title
            if doi not in papers:
                papers[doi or title] = {'doi': doi, 'title': title, 'sources': []}
            papers[doi or title]['sources'].append(name)
    (OUT / 'discovery_index.json').write_text(json.dumps(list(papers.values()), ensure_ascii=False, indent=2), encoding='utf-8')
    src = Path(r'F:\Zotero\storage\47IHA9KL\Maximum likelihood estimation for geolocation in the presence of multipath-Friedlander-2014.pdf')
    with fitz.open(src) as pdf:
        (OUT / 'Friedlander2014.txt').write_text('\n\n'.join(f'PDF PAGE {i+1}\n'+p.get_text() for i,p in enumerate(pdf)), encoding='utf-8')
        for i in [1, 2, 4]:
            pdf[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / f'Friedlander_p{i+1}.png')
    (OUT / 'source_identity.json').write_text(json.dumps({'pdf': str(src),
        'pdf_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
        'type': 'LITERATURE_REVIEW_NO_LOCALIZATION_EXPERIMENT', 'query_count': len(QUERIES),
        'deduplicated_records': len(papers), 'zotero_items': len(inventory)}, indent=2), encoding='utf-8')
    print('DONE', len(papers), 'records;', len(inventory), 'Zotero items')


if __name__ == '__main__':
    main()
