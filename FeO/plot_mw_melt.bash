#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

base='mw_melt_eqm_25GPa'

psbasemap -JX15/10 -R20/140/-2/1 -B20f5:"P (GPa)":/0.5f0.1:"ln D@+melt/mw@+@-FeO@-":SWen -K -P > ${base}.ps

awk 'NR>1' output/mw_melt_Fe_Kd_2773.0K | psxy -J -R -O -K -W1,black,- >> ${base}.ps
awk 'NR>1' output/mw_melt_Fe_Kd_2873.0K | psxy -J -R -O -K -W1,black >> ${base}.ps
awk 'NR>1' output/mw_melt_Fe_Kd_2973.0K | psxy -J -R -O -K -W1,black,- >> ${base}.ps
awk 'NR>1' output/mw_melt_Fe_Kd_3073.0K | psxy -J -R -O -K -W1,black >> ${base}.ps
awk 'NR>1' output/mw_melt_Fe_Kd_3173.0K | psxy -J -R -O -K -W1,black,- >> ${base}.ps

awk '(NR>1) {print $1, log($7*(1-$7)/$5), $1-$2, $1, $1, $1+$2, log(($7-$8)*(1-($7-$8))/($5+$6)), log($7*(1-$7)/$5), log($7*(1-$7)/$5), log(($7+$8)*(1-($7+$8))/($5-$6))}' data/Frost_et_al_2010_fper_melt.dat | psxy -J -R -O -EXY0 -Sc0.1c -W1,blue -K >> ${base}.ps


awk '(NR>1) {print $2, log($7*(100-$7)/$9/10000), log(($7-$8)*(100-($7-$8))/($9+$10)/10000), log($7*(100-$7)/$9/10000), log($7*(100-$7)/$9/10000), log(($7+$8)*(100-($7+$8))/($9-$10)/10000)}' data/Ozawa_et_al_2008_fper_iron.dat | psxy -J -R -O -EY0 -Sc0.1c -W1,red >> ${base}.ps



ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

evince ${base}.pdf
