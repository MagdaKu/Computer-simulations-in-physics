import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def potential(r,m,M,G):
    r_length =  (r[0]**2 + r[1]**2)**0.5
    return -G*m*M/r_length

def kinetic(v,m):
    v =  (v[0]**2 + v[1]**2)**0.5
    return 0.5*m*v**2

def force(r,m,M,G):
    r_norm = np.linalg.norm(r)
    return  -G * m * M / r_norm**3 * r


def euler():
    global m,M,G,t,dt
    r=np.array([2, 0])
    p=np.array([0, 0.1])
    v = p/m
    r_x_array = np.array([r[0]])
    r_y_array = np.array([r[1]])
    E_pot = np.array([potential(r,m,M,G)])
    E_kin = np.array([kinetic(v,m)])
    N = int(t/dt) #steps
    for i in range(N-1):
        F_new = force(r,m,M,G)
        p_new = p + F_new*dt
        v_new = p_new/m
        r_new = r + v_new*dt + 0.5*F_new/m *dt**2 
        r_x_array = np.append(r_x_array, r_new[0])
        r_y_array = np.append(r_y_array, r_new[1])
        E_pot = np.append(E_pot, potential(r_new,m,M,G))
        E_kin = np.append(E_kin, kinetic(v_new,m))
        r = r_new
        v = v_new
        p = p_new
    return r_x_array, r_y_array, E_pot, E_kin


def verlet():
    global m,M,G,t,dt
    r=np.array([2, 0])
    p=np.array([0, 0.1])
    v = p/m
    r_x_array = np.array([r[0]])
    r_y_array = np.array([r[1]])
    E_pot = np.array([potential(r,m,M,G)])
    E_kin = np.array([kinetic(v,m)])
    r0 = r - v*dt
    N = int(t/dt) #steps
    for i in range(N-1):
        F_new = force(r,m,M,G)
        r_new = 2*r - r0 + F_new/m *dt**2 
        v_new = 1/(2*dt) * (r_new - r0)
        r_x_array = np.append(r_x_array, r_new[0])
        r_y_array = np.append(r_y_array, r_new[1])
        E_pot = np.append(E_pot, potential(r_new,m,M,G))
        E_kin = np.append(E_kin, kinetic(v_new,m))
        r0 = r
        r = r_new
        v = v_new
    return r_x_array, r_y_array, E_pot, E_kin



def leapfrog():
    global m,M,G,t,dt
    r=np.array([2, 0])
    p=np.array([0, 0.1])
    v = p/m
    r_x_array = np.array([r[0]])
    r_y_array = np.array([r[1]])
    E_pot = np.array([potential(r,m,M,G)])
    E_kin = np.array([kinetic(v,m)])
    v0 = v - 1/m *force(r,m,M,G) *0.5 * dt
    N = int(t/dt) #steps
    for i in range(N-1):
        F_new = force(r,m,M,G)
        v1 = v0 + F_new/m *dt
        r_new = r + v1*dt 
        v_new = (v1 + v0)/2
        r_x_array = np.append(r_x_array, r_new[0])
        r_y_array = np.append(r_y_array, r_new[1])
        E_pot = np.append(E_pot, potential(r_new,m,M,G))
        E_kin = np.append(E_kin, kinetic(v_new,m))
        r = r_new
        v0 = v1
    return r_x_array, r_y_array, E_pot, E_kin



def plotting(r_x_array, r_y_array, E_pot, E_kin, method):
    plt.figure()
    plt.title(f"Orbit of the mass m around M, method: {methods[method]}")
    plt.scatter(0,0, c = "green", label = "M")
    plt.plot(r_x_array, r_y_array, label = "m")
    plt.grid()
    plt.legend()
    plt.show()

    N_array = np.arange(0, t, dt)
    plt.figure()
    plt.suptitle(f"Method: {methods[method]}")
    plt.subplot(3,1,1)
    plt.title("Potential energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.plot(N_array, E_pot)

    plt.subplot(3,1,2)
    plt.title("Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.plot(N_array, E_kin)

    plt.subplot(3,1,3)
    plt.title("Total energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.plot(N_array, E_pot + E_kin)

    plt.tight_layout()
    plt.show()



G = 0.01
M = 500
m = 0.1
dt = 0.001
t = 10
methods = ["Euler", "Verlet", "Leapfrog"]

#Euler
r_x_array, r_y_array, E_pot, E_kin = euler()
plotting(r_x_array, r_y_array, E_pot, E_kin, method = 0)

#Verlet
r_x_array, r_y_array, E_pot, E_kin = verlet()
plotting(r_x_array, r_y_array, E_pot, E_kin, method = 1)

#Leapfrog
r_x_array, r_y_array, E_pot, E_kin = leapfrog()
plotting(r_x_array, r_y_array, E_pot, E_kin, method = 2)




#extra - Chenciner’s ballet
G = 1  
m1 = m2 = m3 = 1  

def equations(t, conditions):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = conditions
    
    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    r3 = np.array([x3, y3])
    
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)

    #a1 = F12/m1 + F13/m1
    ax1 = -G * m2 * (x1 - x2) / r12**3 - G * m3 * (x1 - x3) / r13**3
    ay1 = -G * m2 * (y1 - y2) / r12**3 - G * m3 * (y1 - y3) / r13**3
    ax2 = -G * m1 * (x2 - x1) / r12**3 - G * m3 * (x2 - x3) / r23**3
    ay2 = -G * m1 * (y2 - y1) / r12**3 - G * m3 * (y2 - y3) / r23**3
    ax3 = -G * m1 * (x3 - x1) / r13**3 - G * m2 * (x3 - x2) / r23**3
    ay3 = -G * m1 * (y3 - y1) / r13**3 - G * m2 * (y3 - y2) / r23**3
    
    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]


r1 = np.array([0.97000436 ,0.24308753])
r2 = -r1
r3=np.array([0,0])
v3=np.array([-0.93240737,0.86473146])
v1 = -v3/2
v2 = v1


initial_conditions = np.concatenate((r1, r2, r3, v1, v2, v3))

t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 500)
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

x1_sol, y1_sol, x2_sol, y2_sol, x3_sol, y3_sol = solution.y[:6]

# Plotting the trajectories
plt.figure(figsize=(10, 10))
plt.plot(x1_sol, y1_sol, label='Body 1', color='r')
plt.plot(x2_sol, y2_sol, label='Body 2', color='g')
plt.plot(x3_sol, y3_sol, label='Body 3', color='b')
plt.scatter(r1[0], r1[1],s=100, color = 'r' ,zorder = 2)
plt.scatter(r2[0], r2[1],s=100, color = 'g' , zorder = 2)
plt.scatter(r3[0], r3[1],s=100, color = 'b', zorder = 2 )
plt.title("Three-Body Problem: Chenciner’s ballet ")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

