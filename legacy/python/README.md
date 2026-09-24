# Crossflow Simulator

This repository contains an early-stage Python simulator for modelling labelled stream cross-sections in microfluidic crossflow systems.

## Current capabilities

- Create 2D channel cross-sections as labelled grids
- Stack inlet streams vertically or horizontally based on flow-rate fractions
- Infer simple inlet layouts from inlet coordinates
- Apply simple surrogate deformations to stacked cross-sections
- Explore experimental pillar-induced deformation using a lubrication approximation

## Repository structure

src/crossflow_simulator/   core simulator package  
examples/                  runnable example scripts  
tests/                     unit tests  
docs/                      project notes and selected figures  
results/                   generated outputs  

## Start here

1. Read `docs/project_overview.md`
2. Run `examples/basic_stacking_demo.py`
3. Look at `src/crossflow_simulator/cross_section.py`
4. Then look at `src/crossflow_simulator/operators.py`
5. Look at `lubrication.py` only if you are interested in the experimental pillar model

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
