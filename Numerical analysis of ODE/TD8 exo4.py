import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

alpha1=0.0
alpha2=5.5
h=0.01
t0=1
tf=2
vt=np.linspace(t0,tf,int((tf-t0)/h)+1)

f=lambda y,t: np.array([y[1], 2*y[0]**3-6*y[0]-2*t**3])

y0=[0,alpha1]
sol=odeint(f,y0,vt)
plt.plot(vt,sol[:,0])

y0=[0,alpha2]
sol=odeint(f,y0,vt)
plt.plot(vt,sol[:,0])

n=0
E=[]
while abs(alpha1-alpha2)>=10**(-8) and n<=100:
    n=n+1
    y0=[0,alpha1]
    sol=odeint(f,y0,vt)
    g1=sol[-1][0]-2.5


    alphan=(alpha1+alpha2)/2
    y0=[0,alphan]
    sol=odeint(f,y0,vt)
    gn=sol[-1][0]-2.5

    if gn==0:
        n=1000
    elif g1*gn<0:
        alpha2=alphan
        g2=gn
    else:
        alpha1=alphan
        g1=gn
    plt.plot(vt,sol[:,0])
print(alphan)
y0=[0,alphan]
sol=odeint(f,y0,vt)
gn=sol[-1][0]-2.5
plt.plot(vt,sol[:,0])