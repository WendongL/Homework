# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:58:15 2019

@author: 44493
"""

import numpy as np
from scipy.integrate import odeint
import pylab as plt

tf=15

f=lambda y,t:np.array([y[0]*y[1]-q*y[0],-y[0]*y[1]+q*(1-y[1])])
Y0=np.random.rand(15,2)*2
vt=np.linspace(0,tf,500)

q=0.5

plt.axis([-0.1,2,0,2])
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
    x1=np.linspace(0,2,20)
    y1=np.linspace(0,2,20)
    X1,Y1=np.meshgrid(x1,y1)
    DX1,DY1=f([X1,Y1],0)
    N=np.sqrt(DX1**2+DY1**2)
    plt.quiver(X1,Y1,DX1/N,DY1/N,N,angles='xy',scale=30)

    x2=np.linspace(0,2,500)
    y2=np.linspace(0,2,500)
    X2,Y2=np.meshgrid(x2,y2)
    DX2,DY2=f([X2,Y2],0)
    plt.contour(X2,Y2,DX2,0,colors='r',linewidths=2)
    plt.contour(X2,Y2,DY2,0,colors='r',linewidths=2)
    # quand t tend ver l'infini, les trajectoires convergent vers le point stationnaires (0.5,0.5)
    plt.plot(np.linspace(-0.1,2,500),(lambda x:1-x)(np.linspace(-0.1,2,500)))
    plt.title('q='+str(q))
plt.show()

plt.figure()
q=1.5

plt.axis([-0.1,2,0,2])
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
    x1=np.linspace(0,2,20)
    y1=np.linspace(0,2,20)
    X1,Y1=np.meshgrid(x1,y1)
    DX1,DY1=f([X1,Y1],0)
    N=np.sqrt(DX1**2+DY1**2)
    plt.quiver(X1,Y1,DX1/N,DY1/N,N,angles='xy',scale=30)

    x2=np.linspace(0,2,500)
    y2=np.linspace(0,2,500)
    X2,Y2=np.meshgrid(x2,y2)
    DX2,DY2=f([X2,Y2],0)
    plt.contour(X2,Y2,DX2,0,colors='r',linewidths=2)
    plt.contour(X2,Y2,DY2,0,colors='r',linewidths=2)
    # quand t tend ver l'infini, les trajectoires convergent vers le point stationnaires (0.5,0.5)
    plt.plot(np.linspace(-0.1,2,500),(lambda x:1-x)(np.linspace(-0.1,2,500)))
    plt.title('q='+str(q))
plt.show()
# Cette fois les trajectoires convergent vers (0,1)