import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('data/Mookherjee_et_al_2016_albite_GGA.dat', dtype=None, encoding=None)

cols = [d[0] for d in data]
data = np.array([list(d)[1:] for d in data]).T

# Get rid of high pressure data points (albite is WEIRD)
#data = data[1:8]


print(cols)
#print(data)

V = 263.02/data[:,1]*1.e-6 # (g/mol)/(g/cm^3)
P = data[:,2]*1.e9
print(V)

plt.scatter(P/1.e9, V)
plt.show()

f = np.log(V/V[0])
f = P/1.e9 

compliances = []
betas = []
for d in data:

    c11, c22, c33 = d[3:6]
    c12, c13, c23 = d[6:9]
    c14, c15, c16 = d[9:12]
    c24, c25, c26 = d[12:15]
    c34, c35, c36 = d[15:18]
    c45, c46, c56 = d[18:21]
    c44, c55, c66 = d[21:24]

    C = [[c11, c12, c13, c14, c15, c16],
         [c12, c22, c23, c24, c25, c26],
         [c13, c23, c33, c34, c35, c36],
         [c14, c24, c34, c44, c45, c46],
         [c15, c25, c35, c45, c55, c56],
         [c16, c26, c36, c46, c56, c66]]

    S = np.linalg.inv(np.array(C))
    betas.append(np.sum(S[:3,:3]))

    compliances.append(S)

compliances = np.array(compliances)
betas = np.array(betas)

fig = plt.figure()
ax = [fig.add_subplot(2, 3, i) for i in range(1,7)]

ax[0].plot(f, betas/betas[0], label=f'beta_T')

for i in range(6):
    for j in range(i,6):
        if i == j:
            ax[0].plot(f, compliances[:,i,j]/compliances[0,i,j], label=f'{i, j}')
        else:
            ax[i+1].plot(f, compliances[:,i,j]/compliances[0,i,j], label=f'{i, j}')
    ax[i].legend()
plt.show()
