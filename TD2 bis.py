# EXO 4
import numpy as np
from scipy.integrate import odeint
import pylab as plt
plt.ion()
plt.show()

f=lambda y,t: np.array([np.cos(t)*y[0]-np.sin(t)*y[1], np.sin(t)*y[0]+np.cos(t)*y[1]])
Y0=np.array([[1,0],[0,1],[0.4,0.2],[1,1.1]])
vt=np.linspace(0,np.pi*4,200)

for i in range(len(Y0)):
    y0=Y0[i]
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1],marker=1,markersize=5,label="x0={},y0={}".format(y0[0],y0[1]))

    plt.legend(loc='best')


Y0=np.array([1+0j,0+1j,0.4+0.2j,1+1.1j])
g= lambda t,y0: np.exp(-1j*np.exp(1j*t) - np.exp(1j*y0)*y0)
for y0 in Y0:
    sol=g(vt,y0)
    plt.plot(np.real(sol),np.imag(sol),linewidth=1,color='k',label="x0={},y0={}".format(np.real(y0),np.imag(y0)))
    plt.legend(loc='best')