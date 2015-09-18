#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="Fe_FeO_interaction_terms"

psbasemap -JX12/8 -R0/200/-100/100 -Ba50f10:"Pressure (GPa)":/a50f10:"Interaction term (kJ/mol)":SWen -K -P > ${base}.ps

awk '$1!="%" {print $1, $2}' data/Fe_FeO_interaction_parameters_3273.0K.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="%" {print $1, $3}' data/Fe_FeO_interaction_parameters_3273.0K.dat | psxy -J -R -O -K -W0.5,black,- >> ${base}.ps
awk '$1!="%" {print $1, $4}' data/Fe_FeO_interaction_parameters_2273.0K.dat | psxy -J -R -O -K -W0.5,blue >> ${base}.ps
awk '$1!="%" {print $1, $5}' data/Fe_FeO_interaction_parameters_2273.0K.dat | psxy -J -R -O -K -W0.5,blue,- >> ${base}.ps
awk '$1!="%" {print $1, $4}' data/Fe_FeO_interaction_parameters_3273.0K.dat | psxy -J -R -O -K -W0.5,red >> ${base}.ps
awk '$1!="%" {print $1, $5}' data/Fe_FeO_interaction_parameters_3273.0K.dat | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps
awk '$1!="%" {print $1, $6}' data/Fe_FeO_interaction_parameters_3273.0K.dat | psxy -J -R -O -K -W1,black >> ${base}.ps

# Now plot data
awk '$1!="%" {print $1, $2}' data/Fe_FeO_interaction_parameters_observed.dat | psxy -J -R -O -K -Sc0.1c -Gblack -W0.5,black >> ${base}.ps
awk '$1!="%" {print $1, $3}' data/Fe_FeO_interaction_parameters_observed.dat | psxy -J -R -O -K -Sc0.1c -Gwhite -W0.5,black >> ${base}.ps


printf "5 -50 \n10 -50" | psxy -J -R -O -K -W1,black >> ${base}.ps
printf "5 -58 \n10 -62" | psxy -J -R -O -K -W0.5,black >> ${base}.ps
printf "35 -58 \n40 -62" | psxy -J -R -O -K -W0.5,black,- >> ${base}.ps
printf "5 -68 \n10 -72" | psxy -J -R -O -K -W0.5,blue >> ${base}.ps
printf "35 -68 \n40 -72" | psxy -J -R -O -K -W0.5,blue,- >> ${base}.ps
printf "5 -78 \n10 -82" | psxy -J -R -O -K -W0.5,red >> ${base}.ps
printf "35 -78 \n40 -82" | psxy -J -R -O -K -W0.5,red,- >> ${base}.ps
echo "7.5 -90" | psxy -J -R -O -K -Sc0.1c -Gblack -W0.5,black >> ${base}.ps
echo "37.5 -90" | psxy -J -R -O -K -Sc0.1c -Gwhite -W0.5,black >> ${base}.ps

printf "7.5 250" | psxy -J -R -O -K -Sc0.1c -Gblack -W0.5,black >> ${base}.ps

echo "12 -50 Ideal (Komabayashi, 2014)" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 -60 W@-Fe-FeO@-        W@-FeO-Fe@- This study, 3273 K" | pstext -J -R -O -K -F+jLM >> ${base}.ps
echo "12 -70 W@-Fe-FeO@-        W@-FeO-Fe@- Frost et al. (2010), 2273 K" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 -80 W@-Fe-FeO@-        W@-FeO-Fe@- Frost et al. (2010), 3273 K" | pstext -J -R -O -K -F+jLM  >> ${base}.ps
echo "12 -90 W@-Fe-FeO@-        W@-FeO-Fe@- Experimental data" | pstext -J -R -O -F+jLM  >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi

evince ${base}.pdf &
