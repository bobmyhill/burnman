# Still to do
- Use analytical derivatives for dV/dP, dV/dT and dS/dT
- We need something to account for the reactions in ASPECT. Not clear how to do this with a compositionally variable system.
- Possibly we also need to ensure that melt is always stable for numerical reasons. We did this in the ULVZ paper by considering a volatile component that was only stable in the melt. We could do the same thing here if it was necessary?

# Files used for calibrated model
- evaluate.py 
  Runs the models over several profiles. Uses FMSH_melt_model, model_parameters and endmember_eos
- FMSH_melt_model.py 
  A large set of helper functions. Uses model_parameters
- model_parameters.py
  All the parameters used by the model. For components, endmember modifications, 
- endmember_eos.py  
  The modified HP equation of state also used in the lower mantle Dannberg paper.


- FMSH_endmembers_linearised.py   
- FeMg2SiO4_melting_linearised.py   
- solve_simple_FMSH_melting_equations.py

- MSH_endmembers.py   
- MSH_endmembers_linearised.py  
- Mg2SiO4_melting_linearised.py
- Mg2SiO4_melting.py    
- H2MgSiO4_construction.py  
- solve_MSH_melting_equations.py              
                  
- calculate_KD_energies.py    
- solve_KD.py                                       

- check_HSC_convention.py                                  
- test_liquidus_function.py

- print_eos_table.py                      
- test_melt_solution.py

- clone_as_mod_hp.py           
- clone_as_mod_hp_nonideal_H2O.py         



# Benchmarking
- Pitzer_Sterner_water_benchmarks.py  
  Benchmarks for the Pitzer-Sterner equation of state. Uses BurnMan
- Silver_Stolper.py           
  Tests the ax_melt model in FMSH_melt_model (which comes from Silver and Stolper)
  using the data from Myhill et al. 2016
- Sakamaki_et_al_2006_comparison.py 
  Compares volumes and densities to the Sakamaki model.