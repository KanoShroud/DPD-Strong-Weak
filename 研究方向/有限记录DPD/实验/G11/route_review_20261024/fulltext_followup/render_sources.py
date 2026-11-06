"""Read-only PDF extraction/rendering for the two supplied Zotero attachments."""
from pathlib import Path
import hashlib
import json
import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = [('isbi2008', 'UIANZ9FM', 'TX6CWGWW'), ('yeredor2012', 'MSAPJXAX', 'EDPBG6C9')]
records = []
for label, attachment, item in SOURCES:
    files = list((Path('F:/Zotero/storage') / attachment).glob('*.pdf'))
    if len(files) != 1:
        raise RuntimeError(f'Ambiguous PDF: {attachment}')
    source = files[0]
    document = fitz.open(source)
    folder = ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    texts = []
    for index, page in enumerate(document):
        texts.append(f'\n--- PDF PAGE {index + 1} ---\n' + page.get_text(sort=True))
        page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7)).save(folder / f'page_{index+1}.png')
    (folder / 'extracted.txt').write_text(''.join(texts), encoding='utf-8')
    # Original figure crops, preserved as paper evidence rather than project results.
    figure_page = document[3]
    box = figure_page.rect
    proportions = (0.52, 0.12, 0.96, 0.37) if label == 'isbi2008' else (0.51, 0.07, 0.96, 0.30)
    clip = fitz.Rect(proportions[0] * box.width, proportions[1] * box.height,
                     proportions[2] * box.width, proportions[3] * box.height)
    figure_page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip).save(folder / 'evidence_figure.png')
    records.append(dict(label=label, item=item, attachment=attachment, path=str(source),
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), pages=len(document)))
(ROOT / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(records, ensure_ascii=False))
