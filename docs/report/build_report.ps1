$ErrorActionPreference = 'Stop'
$reportDirectory = $PSScriptRoot
$compilerCommand = Get-Command xelatex -ErrorAction SilentlyContinue
if ($compilerCommand) {
    $compilerPath = $compilerCommand.Source
} else {
    $compilerCandidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs/MiKTeX/miktex/bin/x64/xelatex.exe'),
        (Join-Path $env:ProgramFiles 'MiKTeX/miktex/bin/x64/xelatex.exe')
    )
    $compilerPath = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $compilerPath) { throw 'Khong tim thay XeLaTeX. Hay cai MiKTeX va them thu muc bin vao PATH.' }
Push-Location -LiteralPath $reportDirectory
try {
    for ($reportPass = 1; $reportPass -le 3; $reportPass++) {
        & $compilerPath --enable-installer -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "Bien dich bao cao that bai o luot $reportPass" }
    }
    Write-Output (Join-Path $reportDirectory 'main.pdf')
} finally {
    Pop-Location
}
