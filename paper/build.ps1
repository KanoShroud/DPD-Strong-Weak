param([string]$TexBin = '')
$ErrorActionPreference = 'Stop'
# No project Python, experiment results or parent-directory dependencies.
function Find-TexTool([string]$Name) {
    if ($TexBin) { return (Join-Path $TexBin "$Name.exe") }
    $tool = Get-Command $Name -ErrorAction SilentlyContinue
    if ($tool) { return $tool.Source }
    $local = "D:\Software\texlive\2026\bin\windows\$Name.exe"
    if (Test-Path -LiteralPath $local) { return $local }
    throw "Cannot find $Name. Add TeX Live to PATH or pass -TexBin."
}
$xelatex = Find-TexTool 'xelatex'
$bibtex = Find-TexTool 'bibtex'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path build -Force | Out-Null
    & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -recorder -output-directory=build paper_draft.tex
    if ($LASTEXITCODE -ne 0) { throw 'Initial XeLaTeX pass failed.' }
    & $bibtex build/paper_draft
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    foreach ($pass in 1..2) {
        & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -recorder -output-directory=build paper_draft.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed." }
    }
    Copy-Item -LiteralPath build/paper_draft.pdf -Destination paper_draft.pdf -Force
    Write-Host "论文编译完成：$PSScriptRoot\paper_draft.pdf"
}
finally { Pop-Location }
