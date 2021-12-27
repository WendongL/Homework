import numpy as np
from scipy.integrate import odeint
import pylab as plt
plt.ion()
plt.show()

a=0
b=1
M=199
h=(b-a)/(M+1)
X=np.linspace(a,b,M+2,retstep=True)
U0=np.zeros(M+2)
U0[1:-1]=np.sin(np.pi*X[0][1:-1])+np.sin(16*np.pi*X[0][1:-1])+2-5*X[0][1:-1]/2
U0[1:-1]=0
U0[0]=0
U0[-1]=0
plt.plot(X[0],U0,marker=10,markersize=5)

A=(np.eye(M)*(-2)+np.eye(M,k=1)+np.eye(M,k=-1))/h**2
B=np.linspace(0,0,M)
B[-1]=2/h**2



T=3.0
k=0.02
vt,k1=np.linspace(0,T,int((T-0)/k+1),retstep=True)
print(k==k1)
F=lambda y,t:A.dot(y)+B*np.sin(np.pi*t)

Y0=U0[1:-1]
sol=odeint(F,Y0,vt)
for i,ligne in enumerate(sol):
    plt.clf()
    plt.axis([a,b,-1.0,2.5])
    plt.title('$t={0}$'.format(vt[i]))
    plt.plot(X[0][1:-1],ligne,'bo-')
    plt.pause(10**-10)
