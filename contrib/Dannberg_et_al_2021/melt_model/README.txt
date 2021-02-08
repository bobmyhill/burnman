These scripts require the following python packages:
numpy
scipy
python-ternary

Python files:

benchmark_boukare.py
This script plots a set of benchmark figures for solid and liquid endmembers
and solid solutions from the Boukare paper.


boukare_convert_eos.py
This script converts the endmember equations of state in the Boukare dataset
into the HP Modified Tait equation of state used in this study.
The dictionaries were then copied into model_parameters.py


burnman_ulvz_melt_model.py
Uses the burnman equations of state to calculate melting curves and
solid properties under lower mantle conditions.
Not an important file.


check_properties.py
Checks the property averaging implemented in the standalone
eos_and_averaging.py file against the tested routines in burnman.
Prints numbers to stdout which should be very close to zero.


eos_and_averaging.py
Equation of state and averaging functions used in this study.


model_parameters.py
Dictionaries containing the model parameters used in this study.


print_eos_table.py
Prints a table of the equation of state parameters used in this study
in LaTeX format.


PT_tables_generator.py
Generates tables of material properties and prints them to files in
output_data/PT_tables/. Only for ASPECT benchmarking purposes.


pyrolite_partitioning.py
Functions used to calculate modified Mg-Fe partition coefficients in the
CFMASO system. Used by ulvz_melt_model_plot_paper_fig.py.


ternary_pseudobinary_comparison.py
Plots figures comparing the full Boukare model in the ternary FeO-MgO-SiO2
system and the pseudobinary system used by ASPECT.


ulvz_melt_model_plot_paper_fig.py
Uses the functions defined in ulvz_melt_model.py
to plot figures used in the paper.


ulvz_melt_model.py
The baseline melt model for the geodynamic simulations
Contains all necessary functions for computing
phase compositions and thermodynamic properties
at a given pressure, temperature and bulk composition


ulvz_melt_model_w_volatiles.py
The baseline melt model modified to include an additional
volatile phase in the melt.
Outputs two plots illustrating the compositions and amounts
of melt for a given bulk volatile content.
