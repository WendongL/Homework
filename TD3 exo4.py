import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp

plt.ion()
plt.show()
#question 1
plt.axis('scaled')
plt.axis([-2,2,-2,2])

Y0=(np.random.rand(10,2)-0.5)*4
vt=np.linspace(0,50,4001)
vtn=vt[::-1]
eps=0
f=lambda y,t: np.array([-y[1]-eps*y[0]*(y[0]**2+y[1]**2), y[0]-eps*y[1]*(y[0]**2+y[1]**2)])

plt.figure(1)
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
    soln=odeint(f,y0,vtn)
    plt.plot(soln[:,0],soln[:,1])
x=np.linspace(-2,2,16)
y=np.linspace(-2,2,16)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
N=np.sqrt(DX**2+DY**2)
plt.quiver(X,Y,DX/N,DY/N,N,angles='xy')

x=np.linspace(-2,2,501)
y=np.linspace(-2,2,501)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
plt.contour(X,Y,DX,0,colors='r',linewidths=2)
plt.contour(X,Y,DY,0,colors='r',linewidths=2)
#question 2
plt.figure(2)
plt.plot(vt,sol[:,0])

#question 3

eps=1
f=lambda y,t: np.array([-y[1]-eps*y[0]*(y[0]**2+y[1]**2), y[0]-eps*y[1]*(y[0]**2+y[1]**2)])

plt.figure(3)
plt.axis('scaled')
plt.axis([-2,2,-2,2])
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
    soln=odeint(f,y0,vtn)
    plt.plot(soln[:,0],soln[:,1])
x=np.linspace(-2,2,16)
y=np.linspace(-2,2,16)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
N=np.sqrt(DX**2+DY**2)
plt.quiver(X,Y,DX/N,DY/N,N,angles='xy')

x=np.linspace(-2,2,501)
y=np.linspace(-2,2,501)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
plt.contour(X,Y,DX,0,colors='r',linewidths=2)
plt.contour(X,Y,DY,0,colors='r',linewidths=2)
#question 4
plt.figure(4)
plt.plot(vt,sol[:,0])