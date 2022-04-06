import matplotlib.pyplot as plt
import numpy as np
from helpers import OptimizationProblem
from helpers import Simple1
from helpers import Simple2
from helpers import Simple3
from copy import deepcopy


def optimize(f, g, x0, n, count, prob):

    x_hist, f_hist = nesterov(f, g, x0, n, count, .4, .6)

    return x_hist, f_hist

def nesterov(f, g, x0, n, count, alpha, beta):
    x = x0
    x_hist = deepcopy(x)
    f_hist = np.array([f(x)])
    v = np.zeros(len(x0))
    while count() < n:

        g_nest = g(x + beta*v)
        d_nest = -1/np.linalg.norm(g_nest) * g_nest
        v_new = beta*v + alpha*d_nest

        x += v_new
        v = v_new
        alpha *= .8
        
        fval = np.array([f(x)])
        x_hist = np.vstack((x_hist, x))
        f_hist = np.vstack((f_hist, fval))

    # x_best = x
    return x_hist, f_hist


# Contour plots and convergence for simple1
p = Simple1()

# x0 = p.x0()
x0 = np.array([-.5,-1.])
x_hist1,f_hist1 = optimize(p.f, p.g, x0, p.n, p.count, p.prob)
p._reset()
x0 = np.array([0.25,2.])
x_hist2,f_hist2 = optimize(p.f, p.g, x0, p.n, p.count, p.prob)
p._reset()
x0 = np.array([2.,1.])
x_hist3,f_hist3 = optimize(p.f, p.g, x0, p.n, p.count, p.prob)

x = np.linspace(-2, 2, 1000)
y = np.linspace(-4, 4, 1000)
xx, yy = np.meshgrid(x,y,indexing='xy')
z = np.zeros(xx.shape)

for i in range(len(x)):
    for j in range(len(y)):
        z[i,j] = p.f(np.array([xx[i, j],yy[i, j]]))


plt.figure(1)
h = plt.contour(x,y,z, levels=[1,10,30,100,250,400])
plt.xlabel("X1")
plt.ylabel("X2")
plt.plot(x_hist1[:,0], x_hist1[:,1],'r')
plt.plot(x_hist2[:,0], x_hist2[:,1],'r')
plt.plot(x_hist3[:,0], x_hist3[:,1],'r')
plt.savefig('contour.png')

plt.figure(2)
plt.plot(f_hist1)
plt.plot(f_hist2)
plt.plot(f_hist3)
plt.xlabel("Iterations")
plt.ylabel("f(x)")
plt.title('Rosenbrock Convergence')
plt.savefig('rosen.png')

# convergence for simple2
q = Simple2()

x0 = np.array([-.5,-2.])
x_hist1,f_hist1 = optimize(q.f, q.g, x0, q.n, q.count, q.prob)
q._reset()
x0 = np.array([0.25,2.])
x_hist2,f_hist2 = optimize(q.f, q.g, x0, q.n, q.count, q.prob)
q._reset()
x0 = np.array([2.,1.])
x_hist3,f_hist3 = optimize(q.f, q.g, x0, q.n, q.count, q.prob)

plt.figure(3)
plt.plot(f_hist1)
plt.plot(f_hist2)
plt.plot(f_hist3)
plt.xlabel("Iterations")
plt.ylabel("f(x)")
plt.title('Himmelblau Convergence')
plt.savefig('himm.png')


# convergence for simple3
s = Simple3()

x0 = np.array([-.5,-1.,1.,1.])
x_hist1,f_hist1 = optimize(s.f, s.g, x0, s.n, s.count, s.prob)
s._reset()
x0 = np.array([0.25,2.,1.,-1.5])
x_hist2,f_hist2 = optimize(s.f, s.g, x0, s.n, s.count, s.prob)
s._reset()
x0 = np.array([2.,1.,-1.5,-2.])
x_hist3,f_hist3 = optimize(s.f, s.g, x0, s.n, s.count, s.prob)

plt.figure(4)
plt.plot(f_hist1)
plt.plot(f_hist2)
plt.plot(f_hist3)
plt.xlabel("Iterations")
plt.ylabel("f(x)")
plt.title('Powell Convergence')
plt.savefig('powell.png')



