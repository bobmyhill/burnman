# Still to do
- Incorporate compositional derivatives into dV/dP, dV/dT and dS/dT in one_phase_eqm and two_phase_eqm (in FMSH_melt_model.py)
- Calculate the bulk properties in evaluate (in evaluate.py)
- Calculate the enthalpy in evaluate? We need something to account for the reactions in ASPECT. Not clear how to do this.
- Possibly we also need to ensure that melt is always stable for numerical reasons. We did this in the ULVZ paper by considering a volatile component that was only stable in the melt. We could do the same thing here if it was necessary?
