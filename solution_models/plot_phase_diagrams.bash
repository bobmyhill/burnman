#!/bin/bash

gmtset FONT_ANNOT_PRIMARY		= 8,4,black
gmtset FONT_ANNOT_SECONDARY		= 10,4,black
gmtset FONT_LABEL       		= 10,4,black



# Fe phase diagram
base='Fe_phase_diagram'

psbasemap -JX12/8 -R0/250/300/6000 -Y4c -B50f10:"Pressure (GPa)":/1000f200:"Temperature (K)":SWen -K -P > ${base}.ps

awk '$1!="#" {print $1, $2}' output_data/hcp_liq_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/fcc_liq_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/fcc_hcp_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/bcc_fcc_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/bcc_hcp_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps

echo "5 500 BCC" | pstext -J -R -O -K >> ${base}.ps
echo "15 1500 FCC" | pstext -J -R -O -K >> ${base}.ps
echo "150 2000 HCP" | pstext -J -R -O -K >> ${base}.ps
echo "20 4000 LIQ" | pstext -J -R -O  >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &


# FeO phase diagram
base='FeO_phase_diagram'

psbasemap -JX12/8 -R0/250/300/6000 -Y4c -B50f10:"Pressure (GPa)":/1000f200:"Temperature (K)":SWen -K -P > ${base}.ps

awk '$1!="#" {print $1, $2}' output_data/FeO_B1_liq_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps

echo "100 2000 B1" | pstext -J -R -O -K >> ${base}.ps
echo "20 4000 LIQ" | pstext -J -R -O  >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &


# FeSi phase diagram
base='FeSi_phase_diagram'

psbasemap -JX12/8 -R0/250/300/6000 -Y4c -B50f10:"Pressure (GPa)":/1000f200:"Temperature (K)":SWen -K -P > ${base}.ps

awk '$1!="#" {print $1, $2}' output_data/FeSi_B2_liq_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/FeSi_B20_liq_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps
awk '$1!="#" {print $1, $2}' output_data/FeSi_B20_B2_equilibrium.dat | psxy -J -R -O -K -W0.5,black >> ${base}.ps

echo "70 1500 B2" | pstext -J -R -O -K >> ${base}.ps
echo "15 1500 B20" | pstext -J -R -O -K >> ${base}.ps
echo "20 4000 LIQ" | pstext -J -R -O  >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps
rm ${base}.epsi

open -a Skim.app ${base}.pdf &

