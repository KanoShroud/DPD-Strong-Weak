"""One-time expansion; preserves text and backs up the previous entry."""
from pathlib import Path
import re
import shutil
import json

root = Path(__file__).resolve().parents[1]
entry = root / 'paper_draft.tex'
backup = root / 'archive/before_standalone_20261018'
backup.mkdir(parents=True, exist_ok=False)
for name in ('paper_draft.tex', 'paper_draft.pdf', 'build.ps1'):
    if (root / name).exists():
        shutil.copy2(root / name, backup / name)
expanded = []

def expand(text):
    def replace(match):
        rel = match[1].replace(r'\PaperRoot', '').strip()
        path = (root / rel).with_suffix('.tex').resolve()
        if not path.is_relative_to(root):
            raise ValueError(path)
        expanded.append(str(path.relative_to(root)))
        return '\n% BEGIN '+rel+'\n'+expand(path.read_text(encoding='utf-8'))+'\n% END '+rel+'\n'
    return re.sub(r'\\input\{([^}]+)\}', replace, text)

result = expand(entry.read_text(encoding='utf-8'))
assert not re.search(r'\\(?:input|include)\{', result)
entry.write_text(result, encoding='utf-8')
(root / 'qa/standalone_expansion.json').write_text(json.dumps(dict(
    expanded_files=expanded, scientific_content_changed=False
), ensure_ascii=False, indent=2), encoding='utf-8')
print('Expanded', len(expanded), 'local inputs into', entry)
