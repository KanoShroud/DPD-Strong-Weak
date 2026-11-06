param([switch]$RefreshAssets, [switch]$TemplateEntry)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:replace'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$paperPython = 'D:\Software\anaconda3\envs\PyTorch\python.exe'
$paperTexBin = 'D:\Software\texlive\2026\bin\windows'
$paperRoot = $PSScriptRoot
$paperOutput = Join-Path $paperRoot 'build'
$paperCwd = $paperRoot
$paperEntry = 'cas-sc-sample.tex'
if ($TemplateEntry) {
    $paperCwd = Join-Path (Split-Path $paperRoot -Parent) 'els-cas-templates'
    $paperOutput = Join-Path $paperRoot 'build\template_entry'
}
New-Item -ItemType Directory -Path $paperOutput -Force | Out-Null
Push-Location -LiteralPath $paperCwd
try {
    if ($RefreshAssets) {
        & $paperPython (Join-Path $paperRoot 'scripts\prepare_assets.py') --offline
        if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed.' }
    }
    & "$paperTexBin\xelatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape "-output-directory=$paperOutput" -jobname=paper $paperEntry
    if ($LASTEXITCODE -ne 0) { throw 'Initial XeLaTeX pass failed.' }
    # BibTeX writes only to a relative child path under its working directory.
    Push-Location -LiteralPath $paperRoot
    try {
        $paperAux = [System.IO.Path]::GetRelativePath($paperRoot, (Join-Path $paperOutput 'paper')).Replace('\','/')
        & "$paperTexBin\bibtex.exe" $paperAux
        if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    }
    finally { Pop-Location }
    foreach ($paperPass in 1..2) {
        & "$paperTexBin\xelatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape "-output-directory=$paperOutput" -jobname=paper $paperEntry
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $paperPass failed." }
    }
    if ($TemplateEntry) {
        Copy-Item -LiteralPath (Join-Path $paperOutput 'paper.pdf') -Destination (Join-Path $paperCwd 'cas-sc-sample.pdf') -Force
    } else {
        Copy-Item -LiteralPath (Join-Path $paperOutput 'paper.pdf') -Destination (Join-Path $paperRoot 'paper.pdf') -Force
        & $paperPython (Join-Path $paperRoot 'scripts\validate_draft.py') --check-polish
        if ($LASTEXITCODE -ne 0) { throw 'Manuscript validation failed.' }
        & $paperPython (Join-Path $paperRoot 'scripts\validate_revision.py')
        if ($LASTEXITCODE -ne 0) { throw 'Evidence validation failed.' }
        Copy-Item -LiteralPath (Join-Path $paperRoot 'paper.pdf') -Destination (Join-Path (Split-Path $paperRoot -Parent) 'els-cas-templates\cas-sc-sample.pdf') -Force
    }
    Write-Host "单栏论文编译完成：$paperOutput\paper.pdf"
}
finally { Pop-Location }
