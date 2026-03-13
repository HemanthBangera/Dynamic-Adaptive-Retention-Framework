<# ============================================================================
#  DARS Layer D – Test Runner Script
#  ===========================================================================
#  Usage:
#      .\tests\run_tests.ps1              # Run all tests
#      .\tests\run_tests.ps1 -Unit        # Run unit tests only (offline)
#      .\tests\run_tests.ps1 -Integration # Run integration tests only (needs Qdrant)
#      .\tests\run_tests.ps1 -Verbose     # Run with extra verbose output
# ============================================================================ #>

param(
    [switch]$Unit,
    [switch]$Integration,
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DARS – Dynamic Adaptive Retention Scoring" -ForegroundColor Cyan
Write-Host "  Layer D Test Suite" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "        Run:  python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$PytestArgs = @("-v", "--tb=short")

if ($VerboseOutput) {
    $PytestArgs += "-s"
    $PytestArgs += "--tb=long"
}

if ($Unit) {
    Write-Host "[MODE] Running UNIT tests only (no Qdrant needed)" -ForegroundColor Yellow
    Write-Host ""
    $PytestArgs += "tests/test_schema.py"
    $PytestArgs += "tests/test_embedding.py"
    $PytestArgs += "tests/test_scoring.py"
}
elseif ($Integration) {
    Write-Host "[MODE] Running INTEGRATION tests only (requires Qdrant cloud)" -ForegroundColor Yellow
    Write-Host ""
    $PytestArgs += "tests/test_integration.py"
}
else {
    Write-Host "[MODE] Running ALL tests" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "[RUN]  $VenvPython -m pytest $($PytestArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $VenvPython -m pytest @PytestArgs

$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  ALL TESTS PASSED" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
}
else {
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  SOME TESTS FAILED  (exit code: $ExitCode)" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
}

exit $ExitCode
