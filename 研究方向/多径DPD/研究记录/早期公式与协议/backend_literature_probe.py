"""Save exact DOI metadata responses for this bounded methodological cross-check."""
from datetime import datetime
from urllib.parse import quote
import requests

from geometry_association_gate import ROOT, write


def main():
    out=ROOT/('backend_literature_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    records=[]
    for i,doi in enumerate(['10.1109/JSYST.2020.3005918','10.3390/s18030892',
       '10.23919/EUSIPCO63174.2024.10715317','10.1006/cviu.1999.0832','10.1109/TSP.2022.3173731']):
        endpoint='https://api.crossref.org/works/'+quote(doi,safe='')
        try:
            response=requests.get(endpoint,timeout=30)
            response.raise_for_status()
            raw=response.json()
            write(out/f'crossref_{i+1}.json',raw)
            records.append(dict(doi=doi,endpoint=endpoint,status='ok',file=f'crossref_{i+1}.json',
                title=raw['message'].get('title'),published=raw['message'].get('published')))
        except (requests.RequestException,ValueError) as exc:
            records.append(dict(doi=doi,endpoint=endpoint,status='failed',error=str(exc)))
    write(out/'endpoints.json',records)
    print(out)
    for r in records:
        print(r['doi'],r['status'],r.get('title'))


if __name__=='__main__':
    main()
