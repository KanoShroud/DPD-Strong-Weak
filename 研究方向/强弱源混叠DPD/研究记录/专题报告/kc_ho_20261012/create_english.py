"""Create the English manuscript once from the approved Chinese final edition."""
from pathlib import Path
import hashlib
import json
import re
import shutil

from english_text import TRANSLATIONS

HERE = Path(__file__).resolve().parent
source = HERE / "Latex PDF/document.tex"
out = HERE / "Latex PDF English"
assert not out.exists(), "English manuscript already exists; edit it directly"
manifest = json.loads((HERE / "chinese_final_manifest.json").read_text(encoding="utf-8"))
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
lines = source.read_text(encoding="utf-8").splitlines()
cjk = {i for i, line in enumerate(lines, 1) if re.search(r"[\u4e00-\u9fff]", line)}
assert cjk == set(TRANSLATIONS), {"missing": sorted(cjk-set(TRANSLATIONS)), "extra": sorted(set(TRANSLATIONS)-cjk)}
coverage = []
for i, translated in TRANSLATIONS.items():
    coverage.append({"source_line": i, "chinese_sha256": hashlib.sha256(lines[i-1].encode("utf-8")).hexdigest(),
                     "english": translated})
    lines[i-1] = translated
text = "\n".join(lines)+"\n"
text = text.replace(r"\usepackage[UTF8,scheme=plain,fontset=windows]{ctex}", r"\usepackage{fontspec}")
text = text.replace(r"\linespread{1.18}", r"\linespread{1.12}")
text = text.replace(r"\setlength{\parindent}{2em}", r"\setlength{\parindent}{1.5em}")
assert not re.search(r"[\u4e00-\u9fff]", text)
out.mkdir()
(out / "figures").mkdir()
(out / "document.tex").write_text(text, encoding="utf-8")
shutil.copyfile(HERE / "Latex PDF/build.ps1", out / "build.ps1")
(out / "translation_coverage.json").write_text(json.dumps({"source_sha256": manifest["source_sha256"],
    "translated_lines": len(coverage), "entries": coverage}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"created": str(out), "translated_lines": len(coverage)}))
