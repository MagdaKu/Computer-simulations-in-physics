import numpy as np
import matplotlib.pyplot as plt


#Sierpinski triangle
x = 0
y = 0

m = (0.5, 0, 0, 0.5)
c1 = (0, 0)
c2 = (0.5, 0)
c3 = (0.25, 3**2/4)
x_array = np.array([])
y_array =  np.array([])
iterations = 40000

for i in range(iterations):
    prob = np.random.random()
    if prob < 1/len(m):
        c = c1
    elif prob < 2/len(m):
        c= c2
    else:
        c=c3
    x_new = m[0]*x + m[1]*y + c[0]
    y_new = m[2]*x + m[3]*y + c[1]
    x = x_new
    y = y_new
    x_array = np.append(x_array, x_new)
    y_array = np.append(y_array, y_new)


plt.figure()
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.xticks([])
plt.yticks([])
plt.scatter(x_array,y_array, s=0.5, marker="o", lw=0, color="green")
plt.show()



#Barnsley’s fern
x = 0
y = 0

p_array = np.array([0.73, 0.1, 0.11, 0.03]) 
m_array = np.array([[0.85, 0.04,-0.04, 0.85], [0.2,-0.26, 0.23, 0.22], [-0.15, 0.28, 0.26, 0.24], [ 0.0, 0.0, 0.0, 0.16]])
c_array = np.array([[0, 1.6], [0, 1.6], [0, 0.44], [0, 0.0]])

x_array_2 = np.array([])
y_array_2 =  np.array([])
iterations = 20000

for i in range(iterations):
    prob = np.random.random()
    if prob < p_array[0]:
        m = m_array[0]
        c = c_array[0]
    elif prob < p_array[0] + p_array[1]:
        m = m_array[1]
        c = c_array[1]
    elif prob < p_array[0] + p_array[1] + p_array[2]:
        m = m_array[2]
        c = c_array[2]   
    else:
        m = m_array[3]
        c = c_array[3]

    x_new = m[0]*x + m[1]*y + c[0]
    y_new = m[2]*x + m[3]*y + c[1]
    x = x_new
    y = y_new
    x_array_2 = np.append(x_array_2, x)
    y_array_2 = np.append(y_array_2, y)

plt.figure()
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.xticks([])
plt.yticks([])
plt.scatter(x_array_2,y_array_2, s=0.5, marker="o", lw=0, color="green")
plt.show()


#Task 2
nr_array = np.array([])
r_array = np.arange(13)
for r in range(len(r_array)):
    H, xe, ye= np.histogram2d(x_array, y_array, bins = 2**r)
    Nr = np.sum(H >0)
    nr_array = np.append(nr_array, Nr)

(a, b), V = np.polyfit(r_array[:9],np.log(nr_array[:9]),deg=1,cov=True)
yfit = a * r_array + b


plt.figure()
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.bottom"] = True
plt.title('log(Nr) as a function of r')
plt.plot(r_array, np.log(nr_array),'k.')
plt.plot(r_array, yfit,'r-')
str_ab = 'D = {0:.3}'.format(a/np.log(2))
plt.ylabel("$log(N_r)$")
plt.xlabel("r")
plt.legend(['data', 'fit: ' + str_ab  ],loc='best')
plt.show()
