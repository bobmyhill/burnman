#!/bin/bash

gmtset FONT_ANNOT_PRIMARY		= 8,4,black
gmtset FONT_ANNOT_SECONDARY		= 10,4,black
gmtset FONT_LABEL       		= 10,4,black



# Ferropericlase - melt equilibrium
base='lnKd_mw_melt'

psbasemap -JX12/8 -R0/140/-2/1 -Y4c -B20f5:"Pressure (GPa)":/1f0.2:"ln D@-FeO@-@+melt/mw@+":SWen -K -P > ${base}.ps

sample1d output_data/lnD_melt_mw_3000.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
sample1d output_data/lnD_melt_mw_3500.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black  >> ${base}.ps
sample1d output_data/lnD_melt_mw_4000.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
sample1d output_data/lnD_melt_mw_4500.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black >> ${base}.ps

makecpt -T2500/3500/10 -Cgray > Tpalette.cpt
awk '$1 != "#"' data/Ozawa_et_al_2008_fper_iron.dat | awk '{print $2, log($7/(100. -$7)/$9), $3,  log(($7-$8)/(100. -($7-$8))/($9+$10)), log($7/(100. -$7)/$9), log($7/(100. -$7)/$9), log(($7+$8)/(100. -($7+$8))/($9-$10))}' | psxy -J -R -O -K -Sc0.3c -EY0 -CTpalette.cpt >> ${base}.ps

psscale -D6/-2/12/0.5ch -B200f50:"Temperature (K)": -O  -CTpalette.cpt -Np >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &




# Perovskite - melt equilibrium
base='pv_melt_equilibrium'

psbasemap -JX6/6 -R0/10/0/10 -Y4c -B2f0.5:"O (wt %)":/2f0.5:"Si (wt %)":SWen -K -P > ${base}.ps

sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_25.0_GPa_2773.0_K.dat -T1 -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_25.0_GPa_4273.0_K.dat -T1 -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps

sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_25.0_GPa_2773.0_K.dat -T1 -I0.01 | awk 'NR==500 {print $1+1, $2, "2773"}' | pstext -J -R -O -K >> ${base}.ps
sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_25.0_GPa_4273.0_K.dat -T1 -I0.01 | awk 'NR==500 {print $1+1, $2-1, "4273"}' | pstext -J -R -O -K >> ${base}.ps


psbasemap -JX6/6 -X8 -R0/10/0/10 -B2f0.5:"O (wt %)":/2f0.5:"Si (wt %)":SWen -K -O >> ${base}.ps

sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_100.0_GPa_2773.0_K.dat -T1 -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_100.0_GPa_4273.0_K.dat -T1 -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps

sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_100.0_GPa_2773.0_K.dat -T1 -I0.01 | awk 'NR==500 {print $1+1, $2, "2773"}' | pstext -J -R -O -K >> ${base}.ps
sample1d output_data/metal_Mg0.8Fe0.2SiO3_equilibrium_100.0_GPa_4273.0_K.dat -T1 -I0.01 | awk 'NR==500 {print $1+1, $2-1, "4273"}' | pstext -J -R -O >> ${base}.ps


#awk '$1!="%" {print $2, $3, $2-2, $2, $2, $2+2, $3-$4, $3, $3, $3+$4}' data/Fe_FeO_eutectic.dat | psxy -J -R -O -K -Sc0.3c -EXY0 -Gred >> ${base}.ps
#awk '$1!="%" {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/Fe_FeO_eutectic_temperature.dat | psxy -J -R -O -K -Sc0.3c -EXY0 -Gred >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &


