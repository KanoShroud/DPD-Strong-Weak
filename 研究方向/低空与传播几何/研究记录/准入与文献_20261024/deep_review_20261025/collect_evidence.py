"""Read-only public literature and Zotero inventory capture; no library writes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.parse
import requests
import fitz

ROOT = Path(__file__).resolve().parent
DOIS = [
    '10.1109/TWC.2025.3554614', '10.1109/TAES.2024.3454564',
    '10.1109/TASL.2008.2004533', '10.1109/TSP.2020.3011804',
    '10.1109/TSP.2022.3173731', '10.1109/ACSSC.2014.7094537',
    '10.3390/s18030892', '10.1109/ACCESS.2018.2889958',
    '10.1109/TVT.2020.2968136', '10.1016/j.dib.2025.112086',
    '10.1109/TSP.2022.3215651',
]
PDFS = {
    'datemm2008': 'https://www.iss.uni-stuttgart.de/forschung/publikationen/scheuing_ieeetasl.pdf',
    'mbd2020': 'https://weizmann.elsevierpure.com/files/111313119/ye_IEEE_TSP_identifiabilityConditions_PV2020.pdf',
    'garcia2015': 'https://arxiv.org/pdf/1505.03193',
    'briso2020_author': 'https://arxiv.org/pdf/2502.17444',
    'semiblind2022': 'https://arl.nus.edu.sg/wp-content/uploads/2022/07/A_Semi-Blind_Method_IEEETransactionsOnSignalProcessing2022.pdf',
    'decentralized2026': 'https://discovery.ucl.ac.uk/id/eprint/10214489/1/Tan_Decentralized_Indoor_Direct_Localization_With_Multiple_Wi-Fi_Access_Points.pdf',
}

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def metadata(doi):
    try:
        r = requests.get('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe=''), timeout=30)
        r.raise_for_status()
        data = r.json()['message']
        return {'doi': doi, 'ok': True, 'metadata': data}
    except Exception as exc:
        return {'doi': doi, 'ok': False, 'error': str(exc)}

def pdf(entry):
    name, url = entry
    record = {'id': name, 'url': url, 'ok': False}
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        if not r.content.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        path = ROOT / 'sources' / (name + '.pdf')
        path.write_bytes(r.content)
        doc = fitz.open(path)
        pages = [p.get_text(sort=True) for p in doc]
        (path.with_suffix('.txt')).write_text('\n'.join(f'\n=== PAGE {i+1} ===\n{t}' for i,t in enumerate(pages)), encoding='utf-8')
        record.update(ok=True, pages=len(pages), sha256=hashlib.sha256(r.content).hexdigest(), path=str(path))
    except Exception as exc:
        record['error'] = str(exc)
    return record

def inventory():
    items=[]
    for start in range(0,10000,100):
        r=requests.get('http://127.0.0.1:23119/api/users/0/items', params={'start':start,'limit':100}, timeout=15)
        r.raise_for_status()
        batch=r.json()
        items.extend(batch)
        if len(batch)<100:
            break
    write_json(ROOT/'zotero_inventory.json', items)
    return items

def main():
    (ROOT/'sources').mkdir(exist_ok=True)
    items=inventory()
    with ThreadPoolExecutor(max_workers=6) as pool:
        metas=list(pool.map(metadata,DOIS))
        pdfs=list(pool.map(pdf,PDFS.items()))
    write_json(ROOT/'crossref.json',metas)
    write_json(ROOT/'source_manifest.json',pdfs)
    for m in metas:
        d=m.get('metadata',{})
        matches=[x['key'] for x in items if x.get('data',{}).get('DOI','').lower()==m['doi'].lower()]
        print(json.dumps({'doi':m['doi'],'ok':m['ok'],'title':d.get('title'),'year':d.get('published'),'zotero':matches},ensure_ascii=False))
    print(json.dumps({'pdfs':pdfs,'inventory_items':len(items)},ensure_ascii=False))

if __name__=='__main__':
    main()
