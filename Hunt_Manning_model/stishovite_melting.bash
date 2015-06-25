#!/bin/bash

ps=stv_melting.ps


psbasemap -JX10 -R0/510/2000/8500 -B100f20:"Pressure (GPa)":/1000:"Temperature (K)":SWen -K > ${ps}

# Zhang et al., 1993 (T [K], P [GPa])
# T = 2648.15 + 33*P - 0.12*P*P
echo 2648.15 33 0.12 | awk '{for (p=0; p<110; p=p+1) {print p, $1 + $2*p - $3*p*p}}' | psxy -J -R -W1,red,- -O -K >> ${ps}


# Millot et al., 2015
# T = 1968.5 + 307.8*P^0.485
echo 1968.5 307.8 0.485 | awk '{for (p=0; p<500; p=p+1) {print p, $1 + $2*p^$3}}' | psxy -J -R -W1,black -O -K >> ${ps}

awk 'NR>1 {print $1, $3, $2, $4}' stv_melting.dat | psxy -J -R -O -Exy0 -Sc0.1c >> ${ps}

# 13 GPa
echo 13 307.8 0.485 | awk '{print $2*$3*$1^(1-$3), "K/GPa"}'