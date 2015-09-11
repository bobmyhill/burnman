#!/bin/bash

# Run burnman script
python ./check.py

base="stv"
psbasemap -JX15/10 -R0/900/1.395/1.420 -B200f100:"Temperature (K)":/0.005f0.001:"Volume (kJ/kbar)":SWen -K -P > ${base}.ps
psxy data/stv_volumes.dat -J -R -O -K -W1,red >> ${base}.ps
psxy burnman_stv_volumes.dat  -J -R -O -W1,black >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi
evince ${base}.pdf

base="stv_298K"
psbasemap -JX15/10 -R0./230/0.96/1.42 -B50f10:"Pressure (GPa)":/0.05f0.1:"Volume (kJ/kbar)":SWen -K -P > ${base}.ps
psxy data/stv_volumes_298K.dat -J -R -O -K -W1,red >> ${base}.ps
psxy burnman_stv_volumes_298K.dat  -J -R -O -W1,black >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi
evince ${base}.pdf


base="gr"
psbasemap -JX15/10 -R0/40/12/13 -B10f5:"Pressure (kbar)":/0.2f0.1:"Volume (kJ/kbar)":SWen -K -P > ${base}.ps
psxy data/gr_volume_300.dat -J -R -O -K -W1,red >> ${base}.ps
psxy data/gr_volume_600.dat -J -R -O -K -W1,red >> ${base}.ps
psxy data/gr_volume_800.dat -J -R -O -K -W1,red >> ${base}.ps
psxy data/gr_volume_1000.dat -J -R -O -K -W1,red >> ${base}.ps
psxy burnman_gr_volumes.dat  -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi
evince ${base}.pdf
