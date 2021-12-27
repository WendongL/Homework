import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp

plt.ion()
plt.show()

f= lambda x: x-np.sin(x)-3*np.pi/2
s= sp.optimize.fsolve(f,0)[0]

df=lambda x: 1-np.cos(x)
x=2
r=1
i=0
R=[]
X=[x]
A=[x,x]
Im=[0,f(x)]
while r>=10**(-8) and i<20:
    i=i+1
    xp=x-f(x)/df(x)
    r=abs(xp-x)
    a=abs(xp-s)
    R.append(r)
    X.append(x)
    A.append(x)
    A.append(x)
    Im.append(0)
    Im.append(f(x))

    x=xp

    print('n ={:4d} rel = {:25.20f} abs = {:25.20f}'.format(i,r,a))

nuage=[]
for  i in range(len(R)-1):
    nuage.append([R[i],R[i+1]])
nuage=np.array(nuage)
plt.loglog(nuage[:,0],nuage[:,1],'o')
P=np.polyfit(np.log10(nuage[:,0]),np.log10(nuage[:,1]),1)
plt.loglog(nuage[:,0],10**P[1]*nuage[:,0]**P[0])
plt.legend(['ordre={:.4f}'.format(P[0])])

plt.figure()
plt.clf()

vt=np.linspace(1,6,201)
plt.plot(vt,f(vt))

for i in range(len(A)-1):
    plt.plot(A[i:i+2],Im[i:i+2])
plt.axhline()