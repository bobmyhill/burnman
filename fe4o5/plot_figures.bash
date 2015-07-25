#!/bin/bash

gmtset FONT_ANNOT_PRIMARY		= 12p,4,black
gmtset FONT_ANNOT_SECONDARY		= 14p,4,black
gmtset FONT_LABEL			= 16p,4,black


base=P-fO2

psxy P-fO2.dat -JX15/10 -R0/24/-12/4 -K -P > ${base}.ps

echo "17 -9 Fe" | pstext -J -R -O -K >> ${base}.ps
echo "8 -7 Fe@-1-y@-O" | pstext -J -R -O -K >> ${base}.ps
echo "21 -0.5 Fe@-5@-O@-6@-" | pstext -J -R -O -K >> ${base}.ps
echo "20 2 Fe@-4@-O@-5@-" | pstext -J -R -O -K >> ${base}.ps
echo "2 -4 Fe@-3@-O@-4@-" | pstext -J -R -O -K >> ${base}.ps
echo "6 1 Fe@-2@-O@-3@-" | pstext -J -R -O -K >> ${base}.ps
echo "1.5 -6 Re-ReO@-2@-" | pstext -F+a20+f10 -J -R -O -K >> ${base}.ps
echo "22 -2.65 Mo-MoO@-2@-" | pstext -F+a20+f10 -J -R -O -K >> ${base}.ps
echo "1.7 3 q" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "6.0 3 coe" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "12  3 stv" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "6.0 -11.2 coe" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "12  -11.2 stv" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "2.5 -9.2 fa" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps
echo "12 -5.6 frw" | pstext -F+f10,grey -J -R -O -K >> ${base}.ps

psbasemap -J -R -O -B4f2:"Pressure (GPa)":/4f2:"log@-10@-(@%6%f@%4%O@-2@-)":SWen >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

convert -density 600 ${base}.pdf ${base}.jpg
evince ${base}.pdf


#################################################

base=P-T_pseudosection


awk '{print $0}' TP-pseudosection.dat |  psxy -JX15/10 -R600/1500/0/25 -B200f100:"Temperature (@+\260@+C)":/4f2:"Pressure (GPa)":SWen -K -P > ${base}.ps

#gmtset FONT_ANNOT_PRIMARY               = 6p,4,black
echo "1360 8.5 +melt" | pstext -J -R -O -K >> ${base}.ps

# Legend
printf "600 25 \n900 25 \n900 15 \n600 15" | psxy -X0.25c -Y-0.25c -J -R -O -K -N -L -G240/240/240 -W0.5p,220/220/220 >> ${base}.ps


loc="750 22.3"
echo ${loc} | awk '{printf "%f %f \n %f %f",$1+36, $2+1, $1-36, $2-1}' | psxy -J -R -O -K -N -L -W0.5p,black >> ${base}.ps
echo ${loc} | awk '{printf "%f %f \n %f %f",$1-36, $2+1, $1+36, $2-1}' | psxy -J -R -O -K -N -L -W0.5p,black >> ${base}.ps
echo ${loc} 0 90 | psxy -J -R -O -K -N -Sw0.36c -Gblack >> ${base}.ps
echo ${loc} 90 180 | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
echo ${loc} 180 270 | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
echo ${loc} 270 360 | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
echo ${loc} | psxy -J -R -O -K -Sc0.36c -W0.5p,black >> ${base}.ps

#gmtset FONT_ANNOT_PRIMARY               = 9p,4,black
echo "825 24 Fe@-4@-O@-5@-" | pstext -J -R -O -K >> ${base}.ps
echo "675 24 mt" | pstext -J -R -O -K >> ${base}.ps
echo "675 21 MO@-x@-" | pstext -J -R -O -K >> ${base}.ps
echo "825 21 w\374s" | pstext -J -R -O -K >> ${base}.ps
echo 675 18.1 270 0.2 | psxy -J -R -O -K -N -Sv0.3c+e+je -W0.5p,black -Gblack >> ${base}.ps
echo 675 16.9 90 0.2 | psxy -J -R -O -K -N -Sv0.3c+e+je  -W0.5p,black -Ggrey >> ${base}.ps
echo "725 18.5 Fe@-4@-O@-5@-" | pstext -J -R -F+jLM -O -K >> ${base}.ps
echo "725 16.5 Fe@-3@-O@-4@-" | pstext -J -R -F+jLM -O -K >> ${base}.ps

echo 0 0 | psxy -J -R -O -K -X-0.25c -Y0.25c >> ${base}.ps

awk '{if (NF>4) print $3, $2}' data/fe4o5_expts.dat | psxy -J -R -O -K -N -Sc0.36c -Gwhite >> ${base}.ps
awk '{print $3, $2, $5}' data/fe4o5_expts.dat | grep Fe4O5 | awk '{print $1, $2, 0, 90}' | psxy -J -R -O -K -N -Sw0.36c -Gblack >> ${base}.ps
awk '{print $3, $2, $5}' data/fe4o5_expts.dat | grep Fe3O4 | awk '{print $1, $2, 90, 180}' | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
awk '{print $3, $2, $5}' data/fe4o5_expts.dat | egrep -h "MoO" | awk '{print $1, $2, 180, 270}' | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
awk '{print $3, $2, $5}' data/fe4o5_expts.dat | egrep -h "ReO" | awk '{print $1, $2, 180, 270}' | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
awk '{print $3, $2, $5}' data/fe4o5_expts.dat | grep FeO | awk '{print $1, $2, 270, 360}' | psxy -J -R -O -K -N -Sw0.36c -Ggrey >> ${base}.ps
awk '{if (NF>4) print $3, $2, $1}' data/fe4o5_expts.dat | grep -v "#" | psxy -J -R -O -K -N -Sc0.36c -W0.5p,black >> ${base}.ps
awk '{if (NF>4) print $3, $2, $1}' data/fe4o5_expts.dat | grep "#" | psxy -J -R -O -K -N -Sc0.36c -W0.5p,grey >> ${base}.ps
# Single crystal
#awk '{print $3, $2, $5}' data/fe4o5_expts.dat | grep xtl | awk '{print $1, $2}' | psxy -J -R -O -K -N -Sa0.10c -X0.15c -Y0.15c -Gwhite -W0.5p,black >> ${base}.ps
#echo 0 0 | psxy -J -R -O -K -X-0.15c -Y-0.15c >> ${base}.ps


awk '$7==1 {print $2, $4,270,0.2}' data/Schollenbruch_expts_P_Kono.dat | psxy -J -R -O -K -N -Sv0.3c+e+je -W0.5p,black -Gblack >> ${base}.ps
awk '$7==2 {print $2, $4,90,0.2}' data/Schollenbruch_expts_P_Kono.dat | psxy -J -R -O -K -N -Sv0.3c+e+je -W0.5p,black -Ggrey >> ${base}.ps



echo "630 4.0 w\374s+mt" | pstext -J -R -F+jLT -O -K >> ${base}.ps
echo "630 7.7 Fe@-4@-O@-5@-+mt" | pstext -J -R -F+jLT -O -K >> ${base}.ps
echo "630 11.6 Fe@-4@-O@-5@-+hem" | pstext -J -R -F+jLT -O -K >> ${base}.ps
echo "1100 19.5 Fe@-4@-O@-5@-+ReO@-2@-" | pstext -F+jLT -J -R -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.ps ${base}.epsi

convert -density 600 ${base}.pdf ${base}.jpg
evince ${base}.pdf
