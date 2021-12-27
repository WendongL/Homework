import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

f= lambda x: x-np.sin(x)-3*np.pi/2
g=lambda x:np.sin(x)+3*np.pi/2
s= sp.optimize.fsolve(f,0)[0]

xold=1
x0=2
x=x0
n=0
E=[]

A=[x0,x0]
Im=[x0,g(x0)]
while abs(x-xold)>10**(-8) and n<50:
    xold=x
    x=g(x)
    n=n+1
    err=abs(x-s)
    E.append(err)
    A.append(x)
    A.append(x)
    Im.append(x)
    Im.append(g(x))
    print('n ={:4d} rel = {:25.20f} erreur={:25.20f}'.format(n,abs(x-xold),err))

nuage=[]
for  i in range(len(E)-1):
    nuage.append([E[i],E[i+1]])
nuage=np.array(nuage)
plt.loglog(nuage[1:,0],nuage[1:,1],'o')
P=np.polyfit(np.log10(nuage[:,0]),np.log10(nuage[:,1]),1)
plt.loglog(nuage[1:,0],10**P[1]*nuage[1:,0]**P[0])
plt.legend(['ordre={:20.15f}'.format(P[0])])

plt.figure()
plt.clf()
vt=np.linspace(0,6,201)
plt.plot(vt,g(vt))
plt.plot(vt,vt)
plt.plot(s,s,'o')
for i in range(len(A)-1):
    plt.plot(A[i:i+2],Im[i:i+2])
