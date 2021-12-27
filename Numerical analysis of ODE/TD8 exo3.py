import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

f= lambda x: x-np.sin(x)-3*np.pi/2
s= sp.optimize.fsolve(f,0)[0]
x1=-10
x2=5
n=0
E=[]
f1=f(x1)
f2=f(x2)

while abs(x1-x2)>=10**(-15) and n<=100:
    xn=(x1+x2)/2
    fn=f(xn)
    err=np.abs(xn-s)
    E.append(err)
    if fn==0:
        n=10000
    elif f1*fn<0:
        x2=xn
        f2=fn
    else:
        x1=xn
        f1=fn
    n=n+1
plt.plot(np.arange(len(E)),np.log10(E))
