#!/bin/bash


# Ternary basemap
side=10
maj=10
min=5

szfont=`echo ${side}*2.0 | bc`
x=`gmtmath -Q $side $side 30 COSD MUL SUB =`
y=`gmtmath -Q $side 30 SIND MUL =`

echo "50 110 H@-2@-O-rich starting compositions and hydrous minerals in MSH" | pstext -JX10 -R0/100/0/100 -K -N -P -X5 -Y12 > ternary.ps

echo 0 100 ${maj} | awk '{for (i=$1; i<$2; i=i+$3) {printf ">> \n %f %f \n %f %f \n>> \n %f %f \n %f %f \n>> \n %f %f \n %f %f \n", i/2, i*sqrt(3)/2, 100-i/2, i*sqrt(3)/2, i, 0, i/2, i*sqrt(3)/2, 100-i, 0, 100-i/2, i*sqrt(3)/2}}' | psxy -R -J -O -K -W0.1,grey,- >> ternary.ps

echo 50 96.6 H@-2@-O | pstext -J -R -O -K -N >> ternary.ps
echo -8.6 -5 MgO | pstext -J -R -O -K -N >> ternary.ps
echo 108.6 -5 SiO@-2@- | pstext -J -R -O -K -N >> ternary.ps

# Bottom axis:
psbasemap -R -J -O -K -B${maj}f${min}:""::,%:s >> ternary.ps


# Left tilted axis
echo "gsave -30 rotate" >> ternary.ps
psbasemap -R -J -O -K -B${maj}f${min}:""::,%:w >> ternary.ps

echo "grestore" >> ternary.ps
# Right tilted axis

psxy -R -J -O -K /dev/null -X$x -Y-$y >> ternary.ps
echo "gsave 30 rotate" >> ternary.ps
psbasemap -R -J -O -K -B${maj}f${min}:""::,%:e >> ternary.ps

echo "grestore" >> ternary.ps


psxy -R -J -O -K -X-$x -Y$y /dev/null >> ternary.ps

printf "0 100 \n 58.333 0"  | awk '{print (100-$1)-($2/2), $2*(sqrt (3)/2)}' | psxy -J -R -O -K -N -W1,100/100/100 >> ternary.ps

printf "0 100 \n 50 0"  | awk '{print (100-$1)-($2/2), $2*(sqrt (3)/2)}' | psxy -J -R -O -K -N -W1,grey >> ternary.ps

printf "0 100 \n 66.666 0"  | awk '{print (100-$1)-($2/2), $2*(sqrt (3)/2)}' | psxy -J -R -O -K -N -W1,grey >> ternary.ps

printf "50 50 \n 0 0"  | awk '{print (100-$1)-($2/2), $2*(sqrt (3)/2)}' | psxy -J -R -O -N -W1,blue >> ternary.ps

# NB composition is in XMgO, XH2O


# Now we start with the actual data

out=splines.out

printf "1.0 5100 \n 0.48 1900 \n 0.345 1200" > per.tmp
XMg=1.0
range=`gmtinfo per.tmp -I -C | awk '{print $1 "/" $2}'`
greenspline per.tmp -R${range} -I0.001 | awk '{print ($1*100)*XMg, (1-$1)*100, $2}' X=${XMg} | awk '{print (100-$1)-($2/2), $2*(sqrt (3)/2), $3}'  > out.tmp

