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

    y0=[0,alpha2]
    sol=odeint(f,y0,vt)
    g2=sol[-1][0]-2.5
    e=abs(g2-g1)
    E.append(e)


    alpha3=alpha2
    alpha2=alpha2 - (alpha2-alpha1)*g2/(g2-g1)
    alpha1=alpha3


    print('n={:4d} intervalle={:10.6f} valeur={:25.20f}'.format(n,abs(alpha1-alpha2),  g2))
print(alpha2)


y0=[0,alpha2]
sol=odeint(f,y0,vt)
gn=sol[-1][0]-2.5
plt.plot(vt,sol[:,0])

plt.figure()
plt.clf()

plt.loglog(E[:-1],E[1:],'-1')

P=np.polyfit(np.log10(E[-6:-1]),np.log10(E[-5:]),1)
plt.loglog(E[-6:-1],10**P[1]*E[-6:-1]**P[0])
plt.legend(['ligne brisée','tan={:.4f}'.format(P[0])])