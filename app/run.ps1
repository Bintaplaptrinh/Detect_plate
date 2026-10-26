param([ValidateRange(1, 65535)][int]$Port = 8787)
$ErrorActionPreference = 'Stop'
$projectDirectory = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
& cargo run --release --manifest-path (Join-Path $PSScriptRoot 'Cargo.toml') -- --project $projectDirectory --port $Port
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
