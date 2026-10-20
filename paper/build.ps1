param([string]$TexBin = '',
      [ValidateSet('Chinese','English','All')][string]$Document = 'Chinese')
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
    $documents = @()
    if ($Document -in @('Chinese','All')) { $documents += @('paper_draft','paper_supplement_zh') }
    if ($Document -in @('English','All')) { $documents += @('paper_submission','paper_supplement') }
    foreach ($name in $documents) {
        & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -recorder -output-directory=build "$name.tex"
        if ($LASTEXITCODE -ne 0) { throw "Initial XeLaTeX pass failed: $name" }
        & $bibtex "build/$name"
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed: $name" }
    }
    # Both documents have aux files before cross-document references are resolved.
    foreach ($pass in 1..2) {
        foreach ($name in $documents) {
            & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -recorder -output-directory=build "$name.tex"
            if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed: $name" }
        }
    }
    foreach ($name in $documents) {
        Copy-Item -LiteralPath "build/$name.pdf" -Destination "$name.pdf" -Force
        Write-Host "论文编译完成：$PSScriptRoot\$name.pdf"
    }
}
finally { Pop-Location }
