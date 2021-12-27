# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:49:30 2019

@author: 44493
"""
import numpy as np
from scipy.integrate import odeint
import pylab as plt

f=lambda y,t: np.array([y[1]-y[0]**3+y[0],-y[0]])
y0=np.array([0.5,0.75])
t0=-10
tf=20
pas=2000
vt=np.linspace(t0,tf,pas)

plt.figure()
plt.clf()
plt.axis([-0.01,1.5,-0.01,4.0])
CI=np.random.rand(6,2)
CI[:,0]=(CI[:,0]-0.5)*80+0.5
CI[:,1]=(CI[:,1]-0.5)*80+1.0


for y0 in CI:
    sol = odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
    plt.plot(np.linspace(-5,5,100),(lambda x:x**3-x)(np.linspace(-5,5,100)))


x=np.linspace(0,1.5,10)
y=np.linspace(0,4,10)
X,Y=np.meshgrid(x,y)
[DX,DY]=f([X,Y],3)
N=np.sqrt(DX**2+DY**2)
plt.quiver(x,y,DX/N,DY/N,angles='xy',scale=20,color='r')


x=np.linspace(-10,10,2001)
y=np.linspace(-10,10,2001)
X,Y=np.meshgrid(x,y)
[DX,DY]=f([X,Y],36)

plt.contour(X,Y,DX,0,colors='r',linewidths=2)
plt.contour(X,Y,DY,0,colors='b',linewidths=2)

vt1=np.linspace(0,40,1001)
plt.figure(2)
plt.clf()
Y0=np.array([[0.45,0.05],[0.45,1.95]])


for y0 in Y0:
    sol = odeint(f,y0,vt1)
    plt.plot(sol[:,0],sol[:,1])
    plt.legend('')
    plt.show()

