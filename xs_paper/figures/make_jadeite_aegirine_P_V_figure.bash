#!/bin/bash

gmtset FONT_ANNOT_PRIMARY	   = 10p,4,black
gmtset FONT_LABEL        	   = 14p,4,black
base="jadeite_aegirine_P_V"

psbasemap -JX12/8 -R0/10/55/65 -Ba1f0.5:"Pressure (GPa)":/a2f1:"Volume (cm@+3@+/mol)":SWen -K -P > ${base}.ps

psxy -J -R -O -K data/jadeite_aegirine_P_V.dat >> ${base}.ps

awk '$1!="%" {print $1, $3*1e6, $1-$2, $1, $1, $1+$2, ($3-$4)*1e6, ($3)*1e6, ($3)*1e6, ($3+$4)*1e6}' ../data/jd_ae_PV_data/ae000_PV.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Ggrey -W0.5,black >> ${base}.ps

awk '$1!="%" {print $1, $3*1e6, $1-$2, $1, $1, $1+$2, ($3-$4)*1e6, ($3)*1e6, ($3)*1e6, ($3+$4)*1e6}' ../data/jd_ae_PV_data/ae026_PV.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Ggrey -W0.5,black >> ${base}.ps

awk '$1!="%" {print $1, $3*1e6, $1-$2, $1, $1, $1+$2, ($3-$4)*1e6, ($3)*1e6, ($3)*1e6, ($3+$4)*1e6}' ../data/jd_ae_PV_data/ae065_PV.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Ggrey -W0.5,black >> ${base}.ps

awk '$1!="%" {print $1, $3*1e6, $1-$2, $1, $1, $1+$2, ($3-$4)*1e6, ($3)*1e6, ($3)*1e6, ($3+$4)*1e6}' ../data/jd_ae_PV_data/ae100_PV.dat | psxy -J -R -O -K -EXY0 -Sc0.1c -Ggrey -W0.5,black >> ${base}.ps

tail -1 ../data/jd_ae_PV_data/ae000_PV.dat | awk '{print $1+0.25, $3*1e6+0.3, xtl}' xtl="Jd" | pstext -J -R -O -K >> ${base}.ps

tail -1 ../data/jd_ae_PV_data/ae026_PV.dat | awk '{print $1+0.5, $3*1e6+0.3, xtl}' xtl="Aeg26" | pstext -J -R -O -K >> ${base}.ps
tail -1 ../data/jd_ae_PV_data/ae065_PV.dat | awk '{print $1+0.5, $3*1e6+0.3, xtl}' xtl="Aeg65" | pstext -J -R -O -K >> ${base}.ps
tail -1 ../data/jd_ae_PV_data/ae100_PV.dat | awk '{print $1-0.25, $3*1e6+0.5, xtl}' xtl="Aeg" | pstext -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
rm ${base}.ps
mv ${base}.epsi ${base}.eps

