import numpy as np
import matplotlib.pyplot as plt

n_dim = 2
B = 256.0 #1.0

aa = np.array([1052.15339761, 352.817699062])   # np.random.uniform(size=n_dim)
bb = np.array([0.00056657223796, 0.0018018018018])   #np.random.uniform(size=n_dim)

nx = 101
xx = np.zeros((n_dim, nx), dtype='float')
xx[0] = np.linspace(0.0, B, nx)
xx[1] = B - xx[0]

def util_fun(x, a, b):
    return a * (1 - np.exp(-b * x))

f = np.zeros((n_dim, nx), dtype='float')
for i in range(n_dim):
    f[i] = util_fun(xx[i], aa[i], bb[i])

f_individual = np.zeros((n_dim, nx), dtype='float')
for i in range(n_dim):
    f_individual[i] = util_fun(xx[0], aa[i], bb[i])

V = f.sum(axis=0)

fV,aV = plt.subplots()
aV.plot(xx[0], f_individual.T)
aV.plot(xx[0], V, lw=2.0)
aV.set_xlabel('x')
aV.set_ylabel('f(x)')

# Solve Lagrangian
x_best = np.zeros(n_dim)
bk_sum = np.sum(1.0 / bb[1:])
x_best[0] = 1.0/(1 + bb[0] * bk_sum) * (B + np.log(aa[0] * bb[0]) * bk_sum - np.sum(np.log(aa[1:] * bb[1:]) / bb[1:]))

for i in range(1, n_dim):
    x_best[i] = (x_best[0]*bb[0] - np.log((aa[0]*bb[0])/(aa[i]*bb[i])))/bb[i]

bV = 0.0
for i in range(n_dim):
    bV += util_fun(x_best[i], aa[i], bb[i])
aV.plot(x_best[0], bV, 'kx')
plt.show()



