@echo off
REM Build script for FarmFederate Paper (Windows)
REM Run this after installing MiKTeX or TeX Live

echo ============================================
echo Building FarmFederate Paper
echo ============================================

REM Build main paper
echo.
echo Building main.tex...
pdflatex -interaction=nonstopmode main.tex
if exist main.aux (
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
)

REM Build architecture diagrams
echo.
echo Building architectures.tex...
pdflatex -interaction=nonstopmode architectures.tex

REM Clean auxiliary files
echo.
echo Cleaning auxiliary files...
del /Q *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot 2>nul

echo.
echo ============================================
echo Build complete!
echo ============================================
echo.
echo Output files:
if exist main.pdf echo   - main.pdf (Complete paper)
if exist architectures.pdf echo   - architectures.pdf (Architecture diagrams)
echo.
pause
