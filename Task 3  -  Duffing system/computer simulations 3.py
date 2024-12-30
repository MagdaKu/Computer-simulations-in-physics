import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz(t, xyz):
    x, y, z = xyz
    global ro
    s, r, b = 10, ro, 8/3.
    return [s*(y-x), x*(r-z) - y, x*y - b*z]


def duffing(t,xv):
    x,v = xv
    a,b,c,vi,f = 1, 1, 0.2, 0.213, 0.2
    return [v, b*x - a*x**3 - c*v + f*np.cos(2*np.pi*vi*t)]

def duffing_2(t,xv):
    x,v = xv
    global f
    a,b,c,vi = 1, 1, 0.2, 0.213
    return [v, b*x - a*x**3 - c*v + f*np.cos(2*np.pi*vi*t)]

def duffing_3(t,xv):
    x,v = xv
    global f
    global c
    a,b,vi = 1, 1, 0.213
    return [v, b*x - a*x**3 - c*v + f*np.cos(2*np.pi*vi*t)]


#Task 1
a, b = 0, 40
t = np.linspace(a, b, 4000)
ro_list = [10,15,28]
for i in range(len(ro_list)):
    ro = ro_list[i]
    plt.subplot(3,1,i+1)
    plt.suptitle("Original Lorenz attractor")
    sol1 = solve_ivp(lorenz, [a, b], [1,1,1], t_eval=t)
    plt.plot(sol1.y[0], sol1.y[2], label = ro_list[i])
    plt.title(ro_list[i])
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.grid
    plt.legend()
    plt.tight_layout()
plt.show()


#Task 2

#duffing
a, b = 0, 40
t = np.linspace(a, b, 4000)
sol1 = solve_ivp(duffing, [a, b], [0, 0.15], t_eval=t)

plt.plot(sol1.y[0], sol1.y[1])
plt.title("Evolution of the system in phase space")
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.grid
plt.tight_layout()
plt.show()

plt.subplot(211)
plt.suptitle("Dependence")
plt.plot(sol1.t, sol1.y[0])
plt.grid
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.subplot(212)
plt.plot(sol1.t, sol1.y[1])
plt.grid
plt.ylabel("$v(t)$")
plt.xlabel("$t$")
plt.show()



#Task 3
a, b = 0, 400
t = np.linspace(a, b, 40000)
f_list = np.linspace(0,1,401)
for f in f_list:
    sol1 = solve_ivp(duffing_2, [a, b], [0, 0.15], t_eval=t)
    plt.plot(sol1.y[0][20000:], sol1.y[1][20000:])
    plt.scatter(1, 0)
    plt.title(f"f = {f:.3f}" )
    plt.xlabel("$x$")
    plt.ylabel("$v$")
    plt.grid
    plt.tight_layout()
    #plt.show()
    #plt.savefig("f=" + str(f) + ".png")
    #plt.close()

    #single period: for f =  0.05, 0.015
    #double period: f =  0.3025
    #quadruple period: f =  0.307 
    #chaos from f = 0.45


#Extra
#duffing
a, b = 0, 400
t = np.linspace(a, b, 400)
f = 0.3
c = 0.07
sol1 = solve_ivp(duffing_3, [a, b], [0, 0.15], t_eval=t)
xsol = sol1.y[0][20000:]
vsol= sol1.y[1][20000:]
plt.figure()
plt.scatter(xsol, vsol, s=3, c="b", lw=0, marker ='o', label="points")
plt.show()
            