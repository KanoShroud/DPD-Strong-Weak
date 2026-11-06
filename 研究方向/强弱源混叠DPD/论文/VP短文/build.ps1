param([ValidateSet('Chinese','English','All')][string]$Document='All',
      [string]$TexBin='')
$ErrorActionPreference='Stop'
function Find-TexTool([string]$Name) {
    if ($TexBin) { return (Join-Path $TexBin "$Name.exe") }
    $found=Get-Command $Name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    $local="D:\Software\texlive\2026\bin\windows\$Name.exe"
    if (Test-Path -LiteralPath $local) { return $local }
    throw "Cannot find $Name; specify -TexBin."
}
$xelatex=Find-TexTool 'xelatex'
$bibtex=Find-TexTool 'bibtex'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path build -Force | Out-Null
    $names=@()
    if ($Document -in @('English','All')) { $names+='main' }
    if ($Document -in @('Chinese','All')) { $names+='main_zh' }
    foreach ($name in $names) {
        & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -output-directory=build "$name.tex"
        if ($LASTEXITCODE) { throw "XeLaTeX failed: $name" }
        & $bibtex "build/$name"
        if ($LASTEXITCODE) { throw "BibTeX failed: $name" }
        foreach ($pass in 1..2) {
            & $xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -output-directory=build "$name.tex"
            if ($LASTEXITCODE) { throw "XeLaTeX failed: $name, pass $pass" }
        }
        Copy-Item -LiteralPath "build/$name.pdf" -Destination "$name.pdf" -Force
        Write-Host "Compiled: $PSScriptRoot\$name.pdf"
    }
} finally { Pop-Location }
