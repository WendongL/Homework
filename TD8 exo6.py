import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

g=9.81
v0=2.5
alpha1=np.pi/8
alpha2=np.pi*2/5
f= lambda y,t: np.array([v0*np.cos(alpha),v0*np.sin(alpha)-g*t])
z=lambda x: np.exp(-x**2)

def runge(f,y0,vt,z):
    y=np.asarray(y0).ravel()
    vy=[y0]
    h=vt[1]-vt[0]
    for t in vt[:-1]:
        p1=f(y,t)
        p2=f(y+h*p1/2,t+h/2)
        p3=f(y+h*p2/2,t+h/2)
        p4=f(y+h*p3,t+h)
        y=y+h/6*(p1+2*p2+2*p3+p4)
        vy.append(y)

        if y[1]<=z(y[0]):
            break
    return np.array(vy)

h=0.01
t0=0
tf=10
y0=[0,1.01]
vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
for alpha in [alpha1,alpha2]:
    sol=runge(f,y0,vt,z)
    plt.plot(sol[:,0],sol[:,1])
    plt.plot(np.linspace(-0.1,5,101),z(np.linspace(-0.1,5,101)))

n=0
while abs(alpha1-alpha2)>=10**(-8) and n<=100:
    n=n+1
    alpha=alpha1
    sol=runge(f,y0,vt,z)
    f1=sol[-1][0]-1

    alphan=(alpha1+alpha2)/2
    alpha=alphan
    sol=runge(f,y0,vt,z)
    fn=sol[-1][0]-1

    if fn==0:
        n=1000
    elif f1*fn<0:
        alpha2=alphan
        f2=fn
    else:
        alpha1=alphan
        f1=fn
print(alphan)


sol=runge(f,y0,vt,z)
plt.plot(sol[:,0],sol[:,1])