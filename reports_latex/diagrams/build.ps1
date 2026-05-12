# Build all standalone figures into pdf/
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue

if (-not $latexmk -and -not $pdflatex) {
    Write-Error "No LaTeX toolchain found (latexmk or pdflatex). Install MiKTeX or TeX Live and add the bin directory to PATH."
}

New-Item -ItemType Directory -Force -Path pdf | Out-Null
$texDir = Join-Path $here "tex"
$figures = Get-ChildItem -Path $texDir -Filter "fig_1_*.tex" | Sort-Object Name

foreach ($f in $figures) {
    Write-Host "Building $($f.Name) ..."
    if ($latexmk) {
        latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=pdf $f.FullName
    } else {
        Push-Location $texDir
        try {
            pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../pdf $f.Name
            if ($LASTEXITCODE -ne 0) { throw "pdflatex failed for $($f.Name)" }
            pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../pdf $f.Name
        } finally {
            Pop-Location
        }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$all = Join-Path $here "all_figures.tex"
if (Test-Path $all) {
    Write-Host "Building all_figures.tex (requires pdf/fig_1_*.pdf) ..."
    if ($latexmk) {
        latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=pdf $all
    } else {
        pdflatex -interaction=nonstopmode -halt-on-error -output-directory=pdf $all
        if ($LASTEXITCODE -eq 0) { pdflatex -interaction=nonstopmode -halt-on-error -output-directory=pdf $all }
    }
}

Write-Host "Done. PDFs in $(Join-Path $here 'pdf')"
