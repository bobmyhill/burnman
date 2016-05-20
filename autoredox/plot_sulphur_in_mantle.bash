#!/bin/bash

base="mars_cmb_garnet_composition"


psbasemap -JX12/8 -R0/0.5/0/1 -K -P -B0.1f0.05:"S@-core@- (atomic fraction)":/0.2f0.1:"@;blue;Fe@+3+@+/Fe@-total@- (garnet)@;;":SnW > ${base}.ps

awk '{print $1, $2}' sulphur_in_mantle.dat | psxy -J -R -O -K -W1,blue >> ${base}.ps
awk '{print $1, $3}' sulphur_in_mantle.dat | psxy -J -R -O -K -W1,red >> ${base}.ps


psbasemap -J -R -O -B0.1f0.05:"S@-core@- (atomic fraction)":/0.2f0.1:"@;red;p@-majorite@-@;;":E >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi
rm ${base}.*ps*


base="mars_mantle_sulphur"


psbasemap -JX12/8 -R0/0.5/0/10000 -K -P -B0.1f0.05:"S@-core@- (atomic fraction)":/2000f500:"S in mantle (ppm)":SWen > ${base}.ps

printf "0.3 0 \n 0.3 10000 \n 0.5 10000 \n 0.5 0 \n" | psxy -J -R -O -K -G255/220/220 >> ${base}.ps
echo 0.4 1000 Khan and Connolly, 2008 | pstext -J -R -O -K -F+jCM >> ${base}.ps

printf "0 2400 \n 0 2900 \n 0.5 2900 \n 0.5 2400 \n" | psxy -J -R -O -K -G255/200/200 >> ${base}.ps
echo 0.01 2650 Tuff et al., 2013 | pstext -J -R -O -K -F+jLM >> ${base}.ps



awk '{print $1, $4}' sulphur_in_mantle.dat | psxy -J -R -O -W1,red >> ${base}.ps


ps2epsi ${base}.ps
epstopdf ${base}.epsi


rm ${base}.*ps*
