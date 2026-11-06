"""Document, source identity and offline-statistics QA for the Chinese edition."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from restructure_unknown_gain import scientific_blocks

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
TEX = BASE / 'Latex PDF'
OUT = BASE / 'qa_g10_report'
RUN = ROOT / 'outputs_e2e/unified/e2e_g10/formal_20261013_231138'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})

    source = (TEX / 'document.tex').read_text(encoding='utf-8')

    def expand(s):
        return re.sub(r'\\input\{([^}]+)\}', lambda m: expand((TEX / m[1]).read_text(encoding='utf-8')), s)

    expanded = expand(source)
    labels = re.findall(r'\\label\{([^}]+)\}', expanded)
    check('unique labels', len(labels) == len(set(labels)))
    check('resolved references', set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', expanded)) <= set(labels))
    check('balanced environments', Counter(re.findall(r'\\begin\{([^}]+)\}', expanded)) == Counter(re.findall(r'\\end\{([^}]+)\}', expanded)))
    bib = re.findall(r'\\bibitem\{([^}]+)\}', expanded)
    cited = []
    for group in re.findall(r'\\cite\{([^}]+)\}', expanded):
        for key in group.split(','):
            if key not in cited:
                cited.append(key)
    check('IEEE first citation order', cited == bib)
    for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', expanded):
        check('image ' + image, (TEX / image).is_file())
    previous = json.loads((BASE / 'unknown_gain_migration.json').read_text(encoding='utf-8'))['editions']['zh']
    legacy = scientific_blocks(source.split(r'\appendix', 1)[1])
    migrated = {'eq:2.7', 'eq:2.8', 'eq:2.9', 'eq:2.10', 'eq:2.14'}
    check('all old numerical tables figures and retained equations unchanged',
          legacy == {k: v for k, v in previous['legacy_scientific_blocks'].items() if k not in migrated})
    for name, digest in previous['figure_assets'].items():
        check('old figure hash ' + Path(name).name, sha(BASE / name) == digest)
    protected = json.loads((OUT / 'preserved_manifest.json').read_text(encoding='utf-8'))['protected']
    check('English and Word unchanged', all(sha(BASE / p) == v for p, v in protected.items()))
    contract = json.loads((RUN / 'contract.json').read_text(encoding='utf-8'))
    check('frozen source identity', all(sha(ROOT / p) == v for p, v in contract['sources'].items()))
    data = json.loads((OUT / 'data_validation.json').read_text(encoding='utf-8'))
    check('all saved rows and spatial arrays unchanged', all(sha(RUN / p) == v for p, v in data['sources'].items()))
    check('1248 tasks and 4320 node metrics', data['rows'] == 1248 and data['metric_records'] == 4320 and data['metrics_match_frozen'])
    frame = pd.read_csv(OUT / 'recomputed_metrics.csv')
    ap = frame[(frame.method == 'AP') & (frame.kind == 'mixed')]
    check('AP time trajectory', [int(ap[ap.budget == t].weak20.sum()) for t in [5,15,30,60]] == [57,60,59,58])
    check('AP failures in three layouts', set(ap[(ap.budget == 60) & (ap.weak20 == 0)].group) == {3,7,11})
    check('isolated weak counts', [int(frame[(frame.method == m) & (frame.kind == 'weak') & (frame.budget == 30)].weak20.sum()) for m in ['LS','MUSIC','MVDR','AP','BNOMP','SBL','MULTISTART','GA']] == [11,8,11,10,10,4,10,9])
    check('MUSIC and SBL isolated strong counts', [int(frame[(frame.method == m) & (frame.kind == 'strong') & (frame.budget == 30)].strong20.sum()) for m in ['MUSIC','SBL']] == [24,6])
    main_text = expanded.split(r'\appendix')[0]
    check('no internal identifiers or pending experiment placeholders', not re.search(r'G10|G9-R|outputs_e2e|待实验|结果尚待|十二个|自由复增益更新', main_text))
    log = (TEX / 'compile-pass-3.log').read_text(encoding='utf-8', errors='replace')
    check('no overflow missing glyph unresolved reference or fatal error', not re.search(r'Overfull|Missing character|undefined|multiply defined|Float too large|^!', log, re.M))
    # Algebra only: single-source matching, conditional projection, column scaling.
    rng = np.random.default_rng(20261014)
    errors = []
    for _ in range(12):
        a = rng.normal(size=(4,2)) + 1j*rng.normal(size=(4,2))
        x = rng.normal(size=4) + 1j*rng.normal(size=4)
        q = a / np.linalg.norm(a,axis=0)
        residual = lambda b: np.linalg.norm(x-b@np.linalg.pinv(b)@x)**2
        errors.append(abs(residual(a)-residual(q)))
        errors.append(abs(residual(q[:,:1])-(np.vdot(x,x).real-abs(np.vdot(q[:,0],x))**2)))
        p = np.eye(4)-q[:,:1]@q[:,:1].conj().T
        u=p@q[:,1]
        gain=abs(np.vdot(u,p@x))**2/np.vdot(u,u).real
        errors.append(abs(residual(q[:,:1])-residual(q)-gain))
    check('36 algebraic identities only', max(errors)<1e-10)
    doc = fitz.open(TEX / 'document.pdf')
    for start in range(0,len(doc),9):
        sheet=Image.new('RGB',(1050,1500),'#bbb')
        for j in range(min(9,len(doc)-start)):
            page=doc[start+j]
            pix=page.get_pixmap(matrix=fitz.Matrix(.55,.55),alpha=False)
            image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            image.thumbnail((342,472))
            x,y=(j%3)*350,(j//3)*500
            sheet.paste(image,(x,y+22))
            ImageDraw.Draw(sheet).text((x+8,y+4),str(start+j+1),fill='black')
            page.get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(OUT/f'page_{start+j+1:02}.png')
        sheet.save(OUT/f'contact_{start+1}.png')
    for stem in ['g10_success','g10_maps_15','g10_cdf_weak']:
        im=Image.open(TEX/'figures'/f'{stem}.png').convert('L')
        im.save(OUT/f'{stem}_gray.png')
    result={'checks':checks,'passed':all(c['passed'] for c in checks),'pages':len(doc),
            'algebra_count':len(errors),'algebra_max_abs_error':max(errors),'moved_common_equations':sorted(migrated),
            'model':'deterministic relative amplitude and baseband delay','validation_scope':'offline statistics, algebra and document QA; no localization runs'}
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'pages':len(doc),'failures':[c for c in checks if not c['passed']]},ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
