"""Register the approved Chinese final edition without changing its contents."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
directory = HERE / "Latex PDF"
target = HERE / "chinese_final_manifest.json"
assert not target.exists(), "The final-edition manifest already exists"
files = [directory / "document.tex", directory / "document.pdf",
         *sorted((directory / "figures").glob("*.pdf"))]
assert len(files) == 12
manifest = {
    "status": "USER_APPROVED_CURRENT_FINAL",
    "edition": "Chinese final edition, 2026-10-11, including R3",
    "authorization": "User designated the current Chinese report as the final edition",
    "source_sha256": "d3c536d171e13f5fa49151a7ff096fefa832add2fb127fa2efbc6214f55c1ab3",
    "files": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    "notes": "The PDF was recompiled by the editor after the previous QA. The approved TeX source is unchanged. No duplicate manuscript or backup was created.",
}
assert manifest["files"][str(Path("Latex PDF/document.tex"))] == manifest["source_sha256"]
target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": manifest["status"], "files": len(files)}, ensure_ascii=False))
