#!/bin/bash

gmtset  FONT_ANNOT_PRIMARY 10p,4,black \
	FONT_LABEL 14p,4,black \
        MAP_GRID_CROSS_SIZE_PRIMARY 0.1i \
	MAP_ANNOT_OFFSET_PRIMARY 0.2c

# PHASE RELATIONS
base="fo_wad_rw"
maxT=`grep -v ">>" ${base}_models.xT | gmtinfo -C | awk '{print $4+300}'`

echo "0 0" | psxy -JX12/8 -R0/1/800/${maxT} -K -P > ${base}.ps
psxy ${base}_models.xT -J -R -O -K >> ${base}.ps


awk '$1<0.1' Smyth_fo_data.xT | psxy -J -R -O -K -St0.15c -W0.5,blue -Gwhite >> ${base}.ps
awk '$1<0.1' Litasov_fo_data.xT | psxy -J -R -O -K -Ss0.15c -W0.5,blue -Gwhite >> ${base}.ps
awk '$1<0.1' Ohtani_rw_data.xT | psxy -J -R -O -K -Sc0.15c -W0.5,black -Gwhite >> ${base}.ps
awk '$1<0.1' Litasov_wad_data.xT | psxy -J -R -O -K -Si0.20c -W0.5,red -Gwhite >> ${base}.ps
awk '$1<0.1' Demouchy_wad_data.xT | psxy -J -R -O -K -Sa0.20c -W0.5,red -Gwhite >> ${base}.ps

awk '$1>0.1' Ohtani_rw_data.xT | psxy -J -R -O -K -Sc0.15c -W0.5,black -Gblack >> ${base}.ps
awk '$1>0.1' Litasov_wad_data.xT | psxy -J -R -O -K -Si0.20c -W0.5,red -Gred >> ${base}.ps
awk '$1>0.1' Demouchy_wad_data.xT | psxy -J -R -O -K -Sa0.20c -W0.5,red -Gred >> ${base}.ps

pslegend -J -R -Dx11.9/7.9/5.2/4/TR -O -K << EOF >> ${base}.ps
N 1
S 0.3c t 0.15c -      0.5,blue  0.7c Smyth et al. (2006; fo)
S 0.3c s 0.15c -      0.5,blue  0.7c Litasov et al. (2009; fo)
S 0.3c a 0.20c -      0.5,red    0.7c Demouchy et al. (2005; wad)
S 0.3c i 0.20c -      0.5,red    0.7c Litasov et al. (2007; wad)
S 0.3c c 0.15c -      0.5,black  0.7c Ohtani et al. (2000; rw)
S 0.3c - 0.3c  -      1,grey,-   0.7c K = 0
S 0.3c - 0.3c  -      1,grey     0.7c K = @~\245@~
S 0.3c - 0.3c  -      1,blue,-    0.7c Model forsterite (13 GPa)
S 0.3c - 0.3c  -      1,red,-    0.7c Model wadsleyite (15 GPa)
S 0.3c - 0.3c  -      1,black,-    0.7c Model ringwoodite (20 GPa)
EOF

psbasemap -J -R -B0.2f0.1:"X@-H2O@-":/500f100:"Temperature (@~\260@~C)":SWen -O >> ${base}.ps

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf


# PARTITIONING
base="wad_fo_partitioning_410"

echo "0 0" | psxy -JX12/8 -R900/2100/0.0/4 -K -P > ${base}.ps
psxy wad_fo_partitioning_410.TD -J -R -O -K >> ${base}.ps
psbasemap -J -R -B500f100:"Temperature (@~\260@~C)":/1f0.5:"D@+wad/fo@+":SWn -O -K >> ${base}.ps

#awk 'NR>1 {print $2, $3, $4}' Litasov_fo_wad_data.PTD | psxy -J -R -O -K -Ey0.1 -Sa0.2c -Gblack -W0.5,black >> ${base}.ps

psxy wad_fo_melt_partitioning_410.TD -J -R900/2100/0.0/0.1 -O -K >> ${base}.ps
psbasemap -J -R -B500f100:"Temperature (@~\260@~C)":/0.01f0.005:"D@+solid/melt@+":E -O -K >> ${base}.ps

awk 'NR>1 {print $2, $3, $4}' Demouchy_wad_data.PTD | psxy -J -R -O -K -Ey0 -Sc0.1c -Gred -W0.5,red >> ${base}.ps


pslegend -J -R -Dx11.9/7.9/5.6/4/TR -O << EOF >> ${base}.ps
N 1
S 0.3c - 0.5c  -      1,black      0.7c D@+wad/fo@+
S 0.3c - 0.5c  -      1,blue,-    0.7c D@+fo/melt@+
S 0.3c - 0.5c  -      1,red,-      0.7c D@+wad/melt@+ 
S 0.3c c 0.10c red      0.5,red    0.7c D@+wad/melt@+ (Demouchy et al., 2005)
EOF

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf


base="rw_wad_partitioning_520"

echo "0 0" | psxy -JX12/8 -R900/2100/0.0/1.6 -K -P > ${base}.ps
psxy rw_wad_partitioning_520.TD -J -R -O -K >> ${base}.ps
psbasemap -J -R -B500f100:"Temperature (@~\260@~C)":/0.2f0.1:"D@+rw/wad@+":SWn -O -K >> ${base}.ps

psxy rw_wad_melt_partitioning_520.TD -J -R900/2100/0.0/0.1 -O -K >> ${base}.ps
psbasemap -J -R -B500f100:"Temperature (@~\260@~C)":/0.01f0.005:"D@+solid/melt@+":E -O -K >> ${base}.ps

awk 'NR>1 {print $3, $4/$6, $4/$6*sqrt($5*$5/$4/$4 + $7*$7/$6/$6)}' Litasov_wad_data.PTD | psxy -J -R -O -K -Ey0 -Sc0.1c -Gred -W0.5,red >> ${base}.ps

awk 'NR>1 {print $2, $3, $4}' Ohtani_rw_data.PTD | psxy -J -R -O -K -Ey0 -Sc0.1c -Gblack -W0.5,black >> ${base}.ps

pslegend -J -R -Dx11.9/7.9/5.1/1.5/TR -O << EOF >> ${base}.ps
N 1
S 0.3c - 0.5c  -      1,black      0.7c D@+rw/wad@+
S 0.3c - 0.5c  -      1,red,-      0.7c D@+wad/melt@+ 
S 0.3c - 0.5c  -      1,black,-    0.7c D@+rw/melt@+
S 0.3c c 0.10c red     0.5,red    0.7c D@+wad/melt@+ (Litasov et al., 2011)
S 0.3c c 0.10c black   0.5,black    0.7c D@+rw/melt@+ (Ohtani et al., 2000) 
EOF

ps2epsi ${base}.ps
epstopdf ${base}.epsi

rm ${base}.ps ${base}.epsi
evince ${base}.pdf
