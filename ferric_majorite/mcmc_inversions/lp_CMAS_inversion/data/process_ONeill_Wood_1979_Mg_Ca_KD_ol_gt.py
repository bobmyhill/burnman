import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

d = np.loadtxt('ONeill_Wood_1979_Mg_Ca_KD_ol_gt.png.dat')


xMg_bulk = 0.9
ol_gt_ratio = 2. # as given in experimental methods, assume molar


def comp(prms, xMg_bulk, xCa_gt, lnKD):
    xFe_ol, xFe_gt = prms
    xMg_gt = 1. - xFe_gt - xCa_gt
    xMg_ol = 1. - xFe_ol
    lnKD2 = np.log((xFe_gt/xMg_gt)/(xFe_ol/xMg_ol))

    xMg_bulk2 = (2*xMg_ol + xMg_gt)/(3. - xCa_gt)

    return np.array([xMg_bulk2 - xMg_bulk, lnKD2-lnKD])


output = []
for i in range(len(d)/5):
    if i <= 3:
        T = 1673.15
        c='k'
    elif i<=8:
        T = 1473.15
        c='blue'
    elif i<=15:
        T = 1273.15
        c='red'
    else:
        T = 1173.15
        c='green'
        
    i0 = i*5
    x = d[i0]
    dx = [(d[i0+4][0] - d[i0+3][0])/2., (d[i0+2][1] - d[i0+1][1])/2.]
    xFe_ol, xFe_gt = fsolve(comp, [0.1, 0.1], args=(0.9, x[0], x[1]))
    #xFe_ol1, xFe_gt1 = fsolve(comp, [0.1, 0.1], args=(0.9, x[0]-dx[0], x[1]+dx[1]))
    #print(xFe_ol1 - xFe_ol, xFe_gt1-xFe_gt)
    
    xCa_gt = x[0]
    dxCa_gt = dx[0]

    dxMg_gt = dxCa_gt
    dxFe_gt = dxCa_gt
    dxMg_ol = 0.005
    dxFe_ol = 0.005
    
    
    xMg_gt = 1. - xFe_gt - xCa_gt
    xMg_ol = 1. - xFe_ol
    output.append([3., T, xMg_ol, dxMg_ol, xFe_ol, dxFe_ol, xMg_gt, dxMg_gt, xFe_gt, dxFe_gt, xCa_gt, dxCa_gt])
    #plt.errorbar(x[0], x[1], xerr=dx[0], yerr=dx[1], color=c)

#plt.xlim(0., 0.3)
#plt.ylim(0., 1.2)
#plt.show()

output = np.array(output)[::-1]
np.savetxt(fname='ONeill_Wood_1979_CFMAS_ol_gt.dat', X=output, fmt='%.4f',
           header='P(GPa) T(K)    xMgol  dxMgol xFeol  dxFeol xMggt  dxMggt xFegt  dxFegt xCagt  dxCagt')
