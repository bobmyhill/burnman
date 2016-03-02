'''
# Check correct activity calculation
def rt_activity_b(X, T):
    Q=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))
    po=Q
    pa=1. - X - 0.5*Q
    pb=X - 0.5*Q
    RTlngSi=pa*(1.-pb)*W[0] - pa*po*W[1] + (1.-pb)*po*W[2]
    RTlnaidealSi=constants.gas_constant*T*np.log(pb*(1.-pa))
    
    return np.sqrt(np.exp((RTlngSi + RTlnaidealSi)/(constants.gas_constant*T)))

compositions=np.linspace(0.001, 0.09, 20)
rtactivity_4=np.empty_like(compositions)
rtactivity_6=np.empty_like(compositions)
rtactivity_8=np.empty_like(compositions)
rtactivity_10=np.empty_like(compositions)
for i, X in enumerate(compositions):
    rtactivity_4[i]=rt_activity_b(X, 4000/constants.gas_constant)
    rtactivity_6[i]=rt_activity_b(X, 6000/constants.gas_constant)
    rtactivity_8[i]=rt_activity_b(X, 8000/constants.gas_constant)
    rtactivity_10[i]=rt_activity_b(X, 10000/constants.gas_constant)

fig1 = mpimg.imread('data/a-x_ordering_Holland_Powell_fig3.png')
plt.imshow(fig1, extent=[0,1,0,1], aspect='auto')
plt.plot( compositions, rtactivity_4, linewidth=1, label='4')
plt.plot( compositions, rtactivity_6, linewidth=1, label='6')
plt.plot( compositions, rtactivity_8, linewidth=1, label='8')
plt.plot( compositions, rtactivity_10, linewidth=1, label='10')
plt.title('FeSi ordering')
plt.xlabel("Compositions")
plt.ylabel("Sqrt activity")
plt.legend(loc='upper right')
plt.show()
'''

'''
# Test diopside jadeite
X=0.5
m=1.
n=1.
DeltaH=-6000.
W=[26000., 16000., 16000.] # convergent ordering

temperatures=np.linspace(373.15, 1373.15, 101)
order=np.empty_like(temperatures)
for i, T in enumerate(temperatures):
    order[i]=optimize.fsolve(eqm_order, 0.999*X*2, args=(X, T, m, n, DeltaH, W))

plt.plot( temperatures, order, linewidth=1, label='order')
plt.title('FeSi ordering')
plt.xlabel("Temperature")
plt.ylabel("Order")
plt.legend(loc='upper right')
plt.show()
'''
