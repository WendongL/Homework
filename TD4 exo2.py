import numpy as np
from scipy.integrate import odeint
import pylab as plt
plt.ion()
plt.show()

a=0
b=1
M=99
h=(b-a)/(M+1)
X=np.linspace(a,b,M+2,retstep=True)
U0=np.zeros(M+2)
U0=np.sin(np.pi*X[0])+np.sin(16*np.pi*X[0])
plt.plot(X[0],U0,marker=10,markersize=5)

A=(np.eye(M)*(-2)+np.eye(M,k=1)+np.eye(M,k=-1))/h**2

T=0.1
k=0.001
vt=np.linspace(0,T,int((T-0)/k+1))
F=lambda y,t: A.dot(y)

sol=odeint(F,U0[1:-1],vt)
for i,ligne in enumerate(sol):
    plt.clf()
    plt.axis([a,b,-1.0,2.5])
    plt.title('$t={0}$'.format(vt[i]))
    plt.plot(X[0][1:-1],ligne,marker=2)
    plt.pause(0.1)
