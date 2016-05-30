#!/bin/bash

gmtset FONT_ANNOT_PRIMARY 12,4,black
gmtset FONT_ANNOT_SECONDARY 10,4,black
gmtset FONT_LABEL 14,4,black

base="mars_cmb_garnet_composition"


psbasemap -JX12/8 -R0/0.5/0/1 -K -P -B0.1f0.05:"S@-melt@- (atomic fraction)":/0.2f0.1:"@;blue;Fe@+3+@+/Fe@-total@- (garnet)@;;":SnW > ${base}.ps

awk '{print $1, $2}' sulphur_in_mantle.dat | psxy -J -R -O -K -W1,blue >> ${base}.ps
awk '{print $1, $3}' sulphur_in_mantle.dat | psxy -J -R -O -K -W1,red >> ${base}.ps


psbasemap -J -R -O -B0.1f0.05:"S@-melt@- (atomic fraction)":/0.2f0.1:"@;red;p@-majorite@-@;;":E >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.*ps*


base="mars_mantle_sulphur"


echo 0 0 | psxy -JX12/8 -R0/0.5/0/10000 -K -P > ${base}.ps

printf "0.3 0 \n 0.3 10000 \n 0.5 10000 \n 0.5 0 \n" | psxy -J -R -O -K -G220/220/220 >> ${base}.ps
echo 0.4 9000 Khan and Connolly, 2008 | pstext -J -R -O -K -F+jCM+f10,4,black >> ${base}.ps

printf "0 2400 \n 0 2900 \n 0.5 2900 \n 0.5 2400 \n" | psxy -J -R -O -K -G255/200/200 >> ${base}.ps
echo "0.01 3180 Mars (Tuff et al., 2013)" | pstext -J -R -O -K -F+jLM+f10,4,red >> ${base}.ps


printf "0 180 \n 0 220 \n 0.5 220 \n 0.5 180 \n" | psxy -J -R -O -K -G200/200/255 >> ${base}.ps
echo "0.01 500 Earth (Palme and O'Neill, 2005)" | pstext -J -R -O -K -F+jLM+f10,4,blue >> ${base}.ps


awk '{print $1, $4}' sulphur_in_mantle.dat | psxy -J -R -O -K -W1,red >> ${base}.ps

psbasemap -JX12/8 -R0/0.5/0/10000 -O -B0.1f0.05:"S@-core@- (atomic fraction)":/2000f500:"S in mantle (ppm)":SWen >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi


rm ${base}.*ps*
