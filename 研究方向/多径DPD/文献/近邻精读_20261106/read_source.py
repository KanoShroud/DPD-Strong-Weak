"""Page-bounded PDF text and visual reading helper; never modifies Zotero."""
import argparse
import json
from pathlib import Path
import fitz

HERE=Path(__file__).resolve().parent
CACHE=Path('C:/Users/Administrator/AppData/Local/Temp/dpd_reading_20261106')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('id')
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--end',type=int)
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    row=next(x for x in json.loads((HERE/'sources.json').read_text(encoding='utf-8')) if x['id']==a.id)
    source=row['pdfs'][0]
    doc=fitz.open(source['path'])
    for i in range(a.start-1,min(a.end or len(doc),len(doc))):
        if a.render:
            dest=CACHE/f'{a.id}_p{i+1}.png'
            doc[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(dest)
            print(dest)
        else:
            print(f'===== {a.id} PDF PAGE {i+1} =====\n{doc[i].get_text(sort=False)}')

if __name__=='__main__':main()
