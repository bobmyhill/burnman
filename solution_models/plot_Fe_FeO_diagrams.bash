#!/bin/bash

gmtset FONT_ANNOT_PRIMARY		= 8,4,black
gmtset FONT_ANNOT_SECONDARY		= 10,4,black
gmtset FONT_LABEL       		= 10,4,black



# Fe-FeO solvus
base='Fe_FeO_solvus'

psbasemap -JX12/8 -R0/1/0/30 -Y4c -B0.2:"X@-FeO@-":/5f1:"Pressure (GPa)":SWen -K -P > ${base}.ps

sample1d output_data/Fe_FeO_solvus_2173.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
sample1d output_data/Fe_FeO_solvus_2373.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black  >> ${base}.ps
sample1d output_data/Fe_FeO_solvus_2573.0_K.dat -I0.001 | psxy -J -R -O -K -W0.5,black >> ${base}.ps

makecpt -T2250/2650/10 -Cgray > Tpalette.cpt
awk '$1 != "%"' data/Fe_FeO_solvus.dat | awk '{print $5/100, $2, $3, ($5-$6)/100, ($5)/100, ($5)/100, ($5+$6)/100}' | psxy -J -R -O -K -Sc0.3c -EX0 -CTpalette.cpt >> ${base}.ps
awk '$1 != "%"' data/Fe_FeO_solvus.dat | awk '{print $9/100, $2, $3, ($9-$10)/100, ($9)/100, ($9)/100, ($9+$10)/100}' | psxy -J -R -O -K -Sc0.3c -EX0 -CTpalette.cpt >> ${base}.ps

psscale -D6/-2/12/0.5ch -B100f25:"Temperature (K)": -O  -CTpalette.cpt -Np >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &


# Fe-FeO eutectic
base='Fe_FeO_eutectic'

psbasemap -JX12/8 -R0/330/0/7000 -Y4c -B50f10:"Pressure (GPa)":/1000f200:"Temperature (K)":SWen -K -P > ${base}.ps

awk '$1!="#" {print $1, $2}' output_data/eutectic_TX.dat | sample1d -I1 | psxy -J -R -O -K -W0.5,black >> ${base}.ps

awk '$1!="%" {print $2, $3, $2-2, $2, $2, $2+2, $3-$4, $3, $3, $3+$4}' data/Fe_FeO_eutectic.dat | psxy -J -R -O -K -Sc0.3c -EXY0 -Gred >> ${base}.ps
awk '$1!="%" {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/Fe_FeO_eutectic_temperature.dat | psxy -J -R -O -K -Sc0.3c -EXY0 -Gred >> ${base}.ps
awk '$1!="#" {print $1, $3, $1-$2, $1, $1, $1+$2, $3-$4, $3, $3, $3+$4}' data/Boehler_1993_Fe_FeO_eutectic.dat | psxy -J -R -O -K -Sc0.3c -EXY0 -Gblue >> ${base}.ps
awk '$1!="%" {print $1, $2, $2-$3, $2, $2, $2+$3}' data/Ozawa_et_al_2011_Fe_FeO_phase_stability.dat | psxy -J -R -O -Ss0.3c -EY0 -Gblack >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &

# Liquidus at 50 GPa
base='Fe_FeO_liquidus'

psbasemap -JX12/8 -R0/22.695/2000/4000 -Y4c -B5f1:"O (wt %)":/500f100:"Temperature (K)":SWen -K -P > ${base}.ps
awk '$1!="#" {print $2, $3}' output_data/Fe_liquidus_50.0_GPa.dat  | sample1d -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $2, $3}' output_data/FeO_liquidus_50.0_GPa.dat  | sample1d -I0.01 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk 'NR==2 {printf "0 %f \n 22.695 %f", $3, $3}' output_data/FeO_liquidus_50.0_GPa.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps

awk '($1-50)*($1-50)<25 {print $4, $3}' data/Seagle_2008_low_eutectic_bounds.dat | psxy -J -R -O -K -Sc0.3c -Gblack >> ${base}.ps
awk '($1-50)*($1-50)<25 {print $4, $3}' data/Seagle_2008_high_eutectic_bounds.dat | psxy -J -R -O -K -Sc0.3c -Gblue >> ${base}.ps
awk '($1-50)*($1-50)<25 {print $4, $3}' data/Seagle_2008_low_liquidus_bounds.dat | psxy -J -R -O -K -Sc0.3c -Gorange >> ${base}.ps
awk '($1-50)*($1-50)<25 {print $4, $3}' data/Seagle_2008_high_liquidus_bounds.dat | psxy -J -R -O -Sc0.3c -Gred >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &


# Interaction terms
base='Fe_FeO_interaction'

psbasemap -JX12/8 -R0/50/0/100 -Y4c -B10f2:"Pressure (GPa)":/20f5:"Interaction term (kJ/mol)":SWen -K -P > ${base}.ps

awk '$1 != "#" {print $1, $2/1000}' output_data/interaction_terms_2273.0_K.dat | sample1d -I0.1 | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1 != "#" {print $1, $3/1000}' output_data/interaction_terms_2273.0_K.dat | sample1d -I0.1 | psxy -J -R -O -K -W0.5,black,- >> ${base}.ps

awk '$1 != "#" {print $1, $2/1000}' output_data/interaction_terms_3273.0_K.dat | sample1d -I0.1 | psxy -J -R -O -K -W0.5,red >> ${base}.ps
awk '$1 != "#" {print $1, $3/1000}' output_data/interaction_terms_3273.0_K.dat | sample1d -I0.1 | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps


awk '$1 != "#"' output_data/experimental_interactions.dat | awk '{print $1, $3/1000, $2}' | psxy -J -R -O -K -Sc0.3c -CTpalette.cpt >> ${base}.ps
awk '$1 != "#"' output_data/experimental_interactions.dat | awk '{print $1, $4/1000, $2}' | psxy -J -R -O -K -Sc0.3c -CTpalette.cpt -W-2 >> ${base}.ps


psscale -D6/-2/12/0.5ch -B100f25:"Temperature (K)": -O  -CTpalette.cpt -Np >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &
