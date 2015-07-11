#!/bin/bash

gmtset  FONT_ANNOT_PRIMARY 10p,4,black \
	FONT_LABEL 14p,4,black \
        MAP_GRID_CROSS_SIZE_PRIMARY 0.1i \
	MAP_ANNOT_OFFSET_PRIMARY 0.2c

# PERICLASE
base="periclase"
maxT=`grep -v ">>" ${base}_models.xT | gmtinfo -C | awk '{print $4+100}'`

echo "0 0" | psxy -JX12/8 -R0/1/600/${maxT} -K -P > ${base}.ps
psxy ${base}_models.xT -J -R -O -K >> ${base}.ps
awk '$3=="b" {print $2/100, $1}' data/13GPa_per-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,grey >> ${base}.ps  
awk '$3=="p" {print $2/100, $1}' data/13GPa_per-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps  
awk '$3=="sp" {print $2/100, $1}' data/13GPa_per-H2O.dat | psxy -J -R -O -K -Sc0.05c -W.5,black >> ${base}.ps  
awk '$3=="l" {print $2/100, $1}' data/13GPa_per-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred -W0.5,red >> ${base}.ps  

pslegend -J -R -Dx11.9/7.9/4/4/TR -O -K << EOF >> ${base}.ps
N 1
S 0.3c c 0.1c  -      0.5,grey   0.7c Brucite
S 0.3c c 0.1c  -      0.5,black  0.7c Periclase
S 0.3c c 0.05c -      0.5,black  0.7c Periclase (minor)
S 0.3c c 0.1c  red    0.5,red    0.7c Liquid
S 0.3c - 0.3c  -      1,grey,-   0.7c     K = 0
S 0.3c - 0.3c  -      1,grey    0.7c K = @~\245@~
S 0.3c - 0.3c  -      1,black    0.7c Model phase relations
EOF

psbasemap -J -R -B0.2f0.1:"X@-H2O@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf

# FORSTERITE
base="forsterite"
maxT=`grep -v ">>" ${base}_models.xT | gmtinfo -C | awk '{print $4+100}'`
addT=0
echo "0 0" | psxy -JX12/8 -R0/1/600/${maxT} -K -P > ${base}.ps
psxy ${base}_models.xT -J -R -O -K >> ${base}.ps
awk '$4=="c" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,blue >> ${base}.ps


awk '$4=="l" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred -W0.5,red >> ${base}.ps
awk '$4=="l_davide" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${base}.ps

awk '$4=="e" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,grey >> ${base}.ps
awk '$4=="se" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.05c -W0.5,grey >> ${base}.ps


awk '$4=="f" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps  
awk '$4=="sf" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.05c -W0.5,black >> ${base}.ps  
awk '$4=="f_davide" {print 1-3*$2/100, $1+a}' a=${addT} data/13GPa_fo-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps


pslegend -J -R -Dx11.9/7.9/4/4/TR -O -K << EOF >> ${base}.ps
N 1
S 0.3c c 0.1c  -      0.5,blue   0.7c Chondrodite
S 0.3c c 0.05c  -     0.5,grey   0.7c Enstatite (minor)
S 0.3c c 0.1c  -      0.5,black  0.7c Forsterite
S 0.3c c 0.05c -      0.5,black  0.7c Forsterite (minor)
S 0.3c c 0.1c  red    0.5,red    0.7c Liquid
S 0.3c - 0.3c  -      1,grey,-   0.7c K = 0
S 0.3c - 0.3c  -      1,grey     0.7c K = @~\245@~
S 0.3c - 0.3c  -      1,black,.  0.7c K = K(T)
S 0.3c - 0.3c  -      1,black    0.7c Model phase relations
EOF

psbasemap -J -R -B0.2f0.1:"X@-H2O@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf



# ENSTATITE
base="enstatite"
maxT=`grep -v ">>" ${base}_models.xT | gmtinfo -C | awk '{print $4+100}'`
addT=0
echo "0 0" | psxy -JX12/8 -R0/1/600/${maxT} -K -P > ${base}.ps
psxy ${base}_models.xT -J -R -O -K >> ${base}.ps
awk '$3=="e" {print 1-2*$2/100, $1+a}' a=${addT} data/13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps  
awk '$3=="se" {print 1-2*$2/100, $1+a}' a=${addT} data/13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.05c -W0.5,black >> ${base}.ps  
awk '$3=="e_davide" {print 1-2*$2/100, $1+a}' a=${addT} data/13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps  
awk '$3=="l" {print 1-2*$2/100, $1+a}' a=${addT} data/13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred -W0.5,red >> ${base}.ps
awk '$3=="l_davide" {print 1-2*$2/100, $1+a}' a=${addT} data/13GPa_en-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred >> ${base}.ps 
awk '$3=="l_Yamada" {print 1-2*$2/100, $1}' data/13GPa_en-H2O.dat | psxy -J -R -O -K -Si0.1c -Gblue >> ${base}.ps 

pslegend -J -R -Dx11.9/7.9/5.4/4/TR -O -K << EOF >> ${base}.ps
N 1
S 0.3c c 0.1c  -      0.5,black  0.7c Clinoenstatite
S 0.3c c 0.05c -      0.5,black  0.7c Clinoenstatite (minor)
S 0.3c c 0.1c  red    0.5,red    0.7c Liquid
S 0.3c i 0.1c  blue   0.5,black  0.7c Liquidus (Yamada et al., 2004)
S 0.3c - 0.3c  -      1,grey,-   0.7c K = 0
S 0.3c - 0.3c  -      1,grey     0.7c K = @~\245@~
S 0.3c - 0.3c  -      1,black,.  0.7c K = K(T)
S 0.3c - 0.3c  -      1,black    0.7c Model phase relations
EOF

psbasemap -J -R -B0.2f0.1:"X@-H2O@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf

# STISHOVITE
base="stishovite"
maxT=`grep -v ">>" ${base}_models.xT | gmtinfo -C | awk '{print $4+100}'`
addT=0
echo "0 0" | psxy -JX12/8 -R0/1/600/${maxT} -K -P > ${base}.ps
awk '{if ($1!=">>") { if ($1<0.999) {print $0} } else {print $0}}'  ${base}_models.xT | psxy  -J -R -O -K >> ${base}.ps

awk '$3=="s" {print 1-$2, $1+a}' a=${addT} data/13GPa_SiO2-H2O.dat | psxy -J -R -O -K -Sc0.1c -W0.5,black >> ${base}.ps  
awk '$3=="ss" {print 1-$2, $1+a}' a=${addT} data/13GPa_SiO2-H2O.dat | psxy -J -R -O -K -Sc0.05c -W0.5,black >> ${base}.ps  

awk '$3=="l" {print 1-$2, $1+a}' a=${addT} data/13GPa_SiO2-H2O.dat | psxy -J -R -O -K -Sc0.1c -Gred -W0.5,red >> ${base}.ps

pslegend -J -R -Dx11.9/7.9/4/4/TR -O -K << EOF >> ${base}.ps
N 1
S 0.3c c 0.1c  -      0.5,black  0.7c Stishovite
S 0.3c c 0.05c -      0.5,black  0.7c Stishovite (minor)
S 0.3c c 0.1c  red    0.5,red    0.7c Liquid
S 0.3c - 0.3c  -      1,grey,-   0.7c K = 0
S 0.3c - 0.3c  -      1,grey     0.7c K = @~\245@~
S 0.3c - 0.3c  -      1,black,.  0.7c K = K(T)
S 0.3c - 0.3c  -      1,black    0.7c Model phase relations
EOF

psbasemap -J -R -B0.2f0.1:"X@-H2O@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf
