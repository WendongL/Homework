import numpy as np
from scipy.integrate import odeint, complex_ode
import pylab as plt
import scipy as sp

plt.ion()
plt.show()

a=6
b=-7
c=5
d=-13

f= lambda y,t: np.array([a*y[0]+b*y[1],c*y[0]+d*y[1]])
plt.axis('scaled')
plt.axis([-4,4,-4,4])

x=np.linspace(-4,4,16)
y=np.linspace(-4,4,16)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
N=np.sqrt(DX**2+DY**2)
plt.quiver(X,Y,DX/N,DY/N,N,angles='xy',scale=30)

x=np.linspace(-4,4,501)
y=np.linspace(-4,4,501)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
plt.contour(X,Y,DX,0,colors='r',linewidths=2)
plt.contour(X,Y,DY,0,colors='b',linewidths=2)

Y0=np.random.rand(12,2)*4-2
vt=np.linspace(0,10,501)
vtn=vt[::-1]
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,-1])
    soln=odeint(f,y0,vtn)
    plt.plot(soln[:,0],soln[:,-1])

A=np.array([[a,b],[c,d]])
D,V=sp.linalg.eig(A)
v1=np.real(V[0])
v2=np.real(V[1])
t=np.transpose(np.linspace(-6,6,301))
t1,v1=np.meshgrid(t,v1)
vecteur1=t1*v1
t2,v2=np.meshgrid(t,v2)
vecteur2=t2*v2

plt.plot(vecteur1[0,:],vecteur1[1,:], color='k')
plt.plot(vecteur2[0,:],vecteur2[1,:], color='k')