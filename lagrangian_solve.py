import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

plt.rc('text', usetex=True)

class LagrangeSolver(object):
    def __init__(self, a, b, t, budget):
        self.n_dim = len(a)
        self.a = a
        self.b = b
        self.t = t
        self.B = budget
        self.ab = self.a*self.b
        
    def util_function(self, x, i):
        return self.a[i] * (1 - np.exp(-self.b[i]*(x+self.t[i])))
        
    def full_value(self, xx):
        V = 0.0
        for i in range(self.n_dim):
            V += self.util_function(xx[i], i)
        return V
        
    def find_max(self):
        # Solve Lagrangian
        x_star = np.zeros(self.n_dim)
        ibk_sum = np.sum(1.0 / self.b[1:])
        x_star[0] = 1.0/(1 + self.b[0]*ibk_sum) * (
            B + (np.log(self.ab[0])-self.b[0]*self.t[0])*ibk_sum -
            np.sum(np.log(self.ab[1:])/self.b[1:])) + np.sum(self.t[1:])

        for i in range(1, n_dim):
            x_star[i] = (self.b[0]*(x_star[0]+self.t[0]) - np.log(self.ab[0]/self.ab[i])) / self.b[i]
            
        return x_star
        

# 2D example
n_dim = 2
B = 1.0 # 256.0 #

aa = np.array([1.0, 0.5]) # np.random.uniform(size=n_dim) # np.array([1052.15339761, 352.817699062])   #
bb = np.array([2.0, 5.0]) # np.random.uniform(size=n_dim) # np.array([0.00056657223796, 0.0018018018018])   #
tt = np.array([0.0, 0.0]) # np.random.uniform(size=n_dim) # np.zeros(n_dim)

lgs = LagrangeSolver(aa, bb, tt, B)

nx = 101
xx = np.zeros((nx, n_dim), dtype='float')
xx[:,0] = np.linspace(0.0, B, nx)
xx[:,1] = B - xx[:,0]

individual_utils = np.zeros((nx, n_dim))
for i in range(n_dim):
    individual_utils[:,i] = lgs.util_function(xx[:,0], i)

V = [lgs.full_value(x) for x in xx]

fV,aV = plt.subplots()
hi = aV.plot(xx[:,0], individual_utils)
aV.plot(xx[:,0], V, lw=2.0)
aV.set_xlabel('$x$')
aV.set_ylabel('$f(x)$')

# Solve Lagrangian
x_best = lgs.find_max()
V_best = lgs.full_value(x_best)
aV.plot(x_best[0], V_best, 'kx')
for i in range(n_dim):
    aV.plot([x_best[i], x_best[i]], [0.0, lgs.util_function(x_best[i],i)], color=hi[i].get_color())
aV.plot([x_best[0], x_best[0]], [lgs.util_function(x_best[0],0), V_best], color=hi[1].get_color())
plt.show()


# 3D example! Woo!

n_dim = 3
B3 = 1.0 # 256.0 #

aa3 = np.array([1.0, 0.2, 0.5])   # np.random.uniform(size=n_dim) #
bb3 =  np.array([2.0, 10.0, 4.0])   #np.random.uniform(size=n_dim) #
tt3 = np.zeros(n_dim)

lgs3 = LagrangeSolver(aa3, bb3, tt3, B3)


simplex_corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
tri_object = tri.Triangulation(simplex_corners[:, 0], simplex_corners[:, 1])

refiner = tri.UniformTriRefiner(tri_object)
simplex_mesh = refiner.refine_triangulation(subdiv=6)

def simplex2cart(a, b, c):
    x = .5 * (2 * b + c)
    y = np.sqrt(3) / 2 * c
    return x, y

def cart2simplex(x, y):
    c = y * 2 / np.sqrt(3)
    b = x - c / 2
    a = 1 - b - c
    return a, b, c

xx3 = np.vstack(cart2simplex(simplex_mesh.x, simplex_mesh.y)).T

individual_utils3 = np.zeros((nx, n_dim))
for i in range(xx3.shape[1]):
    individual_utils3[:,i] = lgs3.util_function(xx[:,0], i)

V3 = [lgs3.full_value(x) for x in xx3]

fV2,aV2 = plt.subplots()
hi2 = aV2.plot(xx[:,0], individual_utils3)
aV2.set_xlabel('$x$')
aV2.set_ylabel('$f(x)$')

f3 = plt.figure()
ax3 = f3.add_subplot(111, projection='3d')
h_scatter = ax3.scatter(simplex_mesh.x, simplex_mesh.y, V3, c=V3, s=10)
# h_cont = ax3.tricontourf(simplex_mesh, zz, 10)
# ax3.set_aspect('equal', 'datalim')

# Solve Lagrangian
x_best3 = lgs3.find_max()
V_best3 = lgs3.full_value(x_best3)
for i in range(n_dim):
    aV2.plot([x_best3[i], x_best3[i]], [0.0, lgs3.util_function(x_best3[i],i)], color=hi2[i].get_color())

xs = simplex2cart(x_best3[0], x_best3[1], x_best3[2])
ax3.plot([xs[0]], [xs[1]], [V_best3], 'kx')
ax3.set_title('Booyah!')
ax3.set_zlabel('$f(x)$')
plt.show()

# f3.savefig('3DSimplexSolve.pdf', bbox_inches='tight')
# fV2.savefig('3DSimplexSolveF.pdf', bbox_inches='tight')
