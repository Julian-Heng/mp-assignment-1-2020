#!/usr/bin/env bash

main()
{
    pdflatex "report.tex" && bibtex "report.aux" && pdflatex "report.tex"
}

main
