param([switch]$RefreshAssets)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:replace'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$paperPython = 'D:\Software\anaconda3\envs\PyTorch\python.exe'
$paperTexBin = 'D:\Software\texlive\2026\bin\windows'
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($RefreshAssets) {
        & $paperPython 'prepare_assets.py' --offline
        if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed.' }
    }
    & "$paperTexBin\xelatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape 'main.tex'
    if ($LASTEXITCODE -ne 0) { throw 'Initial XeLaTeX pass failed.' }
    & "$paperTexBin\bibtex.exe" 'main'
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    foreach ($paperPass in 1..2) {
        & "$paperTexBin\xelatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape 'main.tex'
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $paperPass failed." }
    }
    & $paperPython 'validate_draft.py' --check-polish
    if ($LASTEXITCODE -ne 0) { throw 'Manuscript validation failed.' }
    & $paperPython 'validate_revision.py'
    if ($LASTEXITCODE -ne 0) { throw 'Revision numerical/provenance preservation check failed.' }
    Write-Host "PDF已生成：$PSScriptRoot\main.pdf"
}
finally { Pop-Location }
