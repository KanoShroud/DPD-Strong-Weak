$ErrorActionPreference = 'Stop'
$texExecutable = 'D:\Software\texlive\2026\bin\windows\xelatex.exe'
if (-not (Test-Path -LiteralPath $texExecutable)) {
    $texExecutable = (Get-Command xelatex -ErrorAction Stop).Source
}
Push-Location -LiteralPath $PSScriptRoot
try {
    for ($pass = 1; $pass -le 3; $pass++) {
        & $texExecutable -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape document.tex *> "compile-pass-$pass.log"
        if ($LASTEXITCODE -ne 0) {
            Get-Content -Encoding UTF8 -LiteralPath "compile-pass-$pass.log" -Tail 45
            throw "XeLaTeX pass $pass failed."
        }
    }
    Get-Content -Encoding UTF8 -LiteralPath 'compile-pass-3.log' -Tail 12
    Write-Output 'Compiled: document.pdf'
} finally {
    Pop-Location
}
