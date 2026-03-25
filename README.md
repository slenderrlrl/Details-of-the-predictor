Superconducting Transition Temperature Predictor (Ca–Mg–H System)
Overview

This repository provides the symbolic regression model developed to predict the superconducting transition temperature (T_c) of Ca–Mg–H ternary hydrides under high pressure.

Due to the complexity of the analytical expression, the full model is provided here to ensure transparency and reproducibility.

Physical Background

The model is constructed based on physically relevant descriptors, including:

Hydrogen-related structural parameter (h)
Electronic density of states descriptor (M)
External pressure (P)
Coulomb pseudopotential (μ*)

These variables are known to influence electron–phonon coupling and thus superconductivity.

Model Description

The predictor is a symbolic regression expression of the form:

f(h, M, P, μ*) → T_c

The full analytical expression is provided in:

model_expression.txt

A Python implementation is provided in:

model.py
Input Parameters
Symbol	Description	Unit
h	Hydrogen-related structural descriptor	(define clearly)
M	Electronic descriptor (e.g., DOS-related)	states/eV
P	Pressure	GPa
μ*	Coulomb pseudopotential	dimensionless
Usage
1. Requirements
Python 3.x
NumPy
2. Example
from model import predictor

Tc = predictor(h=..., M=..., P=..., mu_star=...)
print(Tc)
Reproducibility

All results reported in the associated paper can be reproduced using the provided model and input data.

Data Availability

The model is archived at:

DOI: 10.5281/zenodo.19212385

Notes
The symbolic expression is intentionally kept in its original form to preserve accuracy.
No additional simplification was applied.
Contact

For questions, please contact:

