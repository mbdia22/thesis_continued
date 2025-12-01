#!/bin/bash

# Create project root directories
mkdir -p paper/sections
mkdir -p src
mkdir -p notebooks
mkdir -p figures

# Create empty files for src
touch src/__init__.py
touch src/models.py
touch src/pricing_rmv.py
touch src/pricing_rfv.py
touch src/utils.py

# Create empty files for paper
touch paper/main.tex
touch paper/references.bib

echo "Project structure created successfully."
