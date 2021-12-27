import numpy as np
import scipy as sp
import pylab as plt
from scipy.interpolate import interp1d

plt.ion()
plt.show()
plt.figure()
plt.clf()

P=np.array([50,-195,58,231,0])/72
vx=np.linspace(-1.0, 3.0, 501)
vy=np.polyval(P,vx)
pts=[-1, 0, 1, 3/2, 3]
ypts=np.polyval(P,pts)

plt.plot(vx,vy,label='Lagrange')
plt.plot(pts,ypts,'o')

P=interp1d(pts,ypts,kind='linear')
vy=P(vx)
plt.plot(vx,vy,label='linear')

P=interp1d(pts,ypts,kind='nearest')
vy=P(vx)
plt.plot(vx,vy,label='nearest')

P=interp1d(pts,ypts,kind='cubic')
vy=P(vx)
plt.plot(vx,vy,label='cubic')

plt.legend(loc='best')
##
A=sp.linalg.solve(np.vander(pts), ypts)
plt.figure()
plt.clf()
plt.subplot(2,1,1)
P=np.array([50,-195,58,231,0])/72
vx=np.linspace(-1.0, 3.0, 501)
vy1=np.polyval(P,vx)
vy2=np.polyval(A,vx)
plt.plot(vx,vy1-vy2)


Q=np.polyfit(pts,ypts,4)
plt.subplot(2,1,2)
vy3=np.polyval(Q,vx)
plt.plot(vx,vy1-vy3)

## exo2
plt.figure()
plt.clf()
f=lambda x: 1/(x**2+0.1)
for n in range(1,7):
    plt.subplot(3,2,n)
    vx=np.linspace(-1,1,n+1)
    vy=f(vx)
    plt.plot(vx,vy,'bo')
    P=np.polyfit(vx,vy,n)
    vx=np.linspace(-1,1,501)
    vy=np.polyval(P,vx)
    plt.plot(vx,vy)

plt.figure()
plt.clf()
for n in range(19,21):
    plt.subplot(1,2,n-18)
    vx=np.linspace(-1,1,n+1)
    vy=f(vx)
    plt.plot(vx,vy,'bo')
    P=np.polyfit(vx,vy,n)
    vx=np.linspace(-1,1,501)
    vy=np.polyval(P,vx)
    plt.plot(vx,vy)

## exo3
plt.figure()
plt.clf()
g=lambda x,n: np.cos((1+2*x)*np.pi/2/(n+1))
f=lambda x: 1/(x**2+0.1)
for n in range(1,7):
    plt.subplot(3,2,n)
    vx=g(np.linspace(0,n+1),n)
    vy=f(vx)
    plt.plot(vx,vy,'bo')
    P=np.polyfit(vx,vy,n)
    vx=np.linspace(-1,1,501)
    vy=np.polyval(P,vx)
    plt.plot(vx,vy)

plt.figure()
plt.clf()
for n in range(19,21):
    plt.subplot(1,2,n-18)
    vx=g(np.linspace(0,n+1),n)
    vy=f(vx)
    plt.plot(vx,vy,'bo')
    P=np.polyfit(vx,vy,n)
    vx=np.linspace(-1,1,501)
    vy=np.polyval(P,vx)
    plt.plot(vx,vy)


