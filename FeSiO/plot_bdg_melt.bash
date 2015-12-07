#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black

base='bdg_melt_eqm_25GPa'

psbasemap -JX10/10 -R0/9.99999/0/10 -B2f1:"O (wt %)":/2f1:"Si (wt %)":SWen -K > ${base}.ps

awk 'NR>1' output/25.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_2773.0K | psxy -J -R -O -K -W1,black >> ${base}.ps
awk 'NR>1' output/25.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_3273.0K | psxy -J -R -O -K -W1,black,- >> ${base}.ps
awk 'NR>1' output/25.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_3773.0K | psxy -J -R -O -K -W1,black >> ${base}.ps
awk 'NR>1' output/25.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_4273.0K | psxy -J -R -O -K -W1,black,- >> ${base}.ps

echo 9 9 25 GPa | pstext -J -R -O -K -F+jRM >> ${base}.ps

psbasemap -JX10/10 -X10c -R0/10/0/10 -B2f1:"O (wt %)":/2f1:"Si (wt %)":Sen -P -O -K >> ${base}.ps

awk 'NR>1' output/120.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_2773.0K | psxy -J -R -O -K -W1,red >> ${base}.ps
awk 'NR>1' output/120.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_3273.0K | psxy -J -R -O -K -W1,red,- >> ${base}.ps
awk 'NR>1' output/120.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_3773.0K | psxy -J -R -O -K -W1,red >> ${base}.ps
awk 'NR>1' output/120.0GPa_0.1_FeSiO3_bdg_melt_wt_O_Si_4273.0K | psxy -J -R -O -K -W1,red,- >> ${base}.ps

echo 9 9 120 GPa | pstext -J -R -O -F+jRM >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

evince ${base}.pdf
