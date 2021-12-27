import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve
from scipy.interpolate import interp1d
from scipy.integrate import quad

plt.ion()
plt.show()

def multipas(f,df,y0,vt):
    y1=np.asarray(y0).ravel()
    h=vt[1]-vt[0]
    vy=[y1]
    t0=vt[0]
    p1=f(y1,t0)
    p2=f(y1+h/2*p1,t0+h/2)
    p3=f(y1+h/2*p2,t0+h/2)
    p4=f(y1+h*p3,t0+h)
    y2=y1+h/6*(p1+2*p2+2*p3+p4)
    vy.append(y2)
    dim=np.size(y0)


    for t in vt[2:]:
        r=1
        i=0

        z=y2

        while r>=10**(-12) and i<10:
            i=i+1
            DF=np.eye(dim)-h/3*df(z,t)
            F=z-y1-h/3*(f(y1,t-2*h)+4*f(y2,t-h)+f(z,t))
            zp=z-solve(DF,F)
            r=sp.linalg.norm(zp-z, np.inf)
            z=zp
            if t <vt[4]:
                print('{:3d} {:20.15f}'.format(i,r))

        vy.append(z)
        y1=y2
        y2=z
    return np.array(vy)


f=lambda y,t: -np.sin(t)*y**2 + 2* np.tan(t)*y
df=lambda y,t: -np.sin(t)*y*2 + 2* np.tan(t)
g=lambda t: 1/np.cos(t)
t0=np.pi/6
tf=np.pi/3
y0=2/np.sqrt(3)
nuage=[]
H=2.0**-np.arange(3,12)*np.pi/3
for h in H:
    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
    vy=multipas(f,df,y0,vt)[:,0]
    vyexacte=g(vt)
    err=np.linalg.norm(vyexacte-vy, np.inf)
    nuage.append([h,err])

nuage=np.array(nuage)
plt.loglog(nuage[:,0],nuage[:,1],'o')
P=np.polyfit(np.log10(nuage[:,0]),np.log10(nuage[:,1]),1)
plt.loglog(nuage[:,0],10**P[1]*nuage[:,0]**P[0],label='ordre={:.4f}'.format(P[0]))
plt.legend(loc='best')


f=lambda y,t: np.array([y[0]-y[0]*y[1], 2*y[0]*y[1]-3/2*y[1]])
df=lambda y,t: np.array([[1-y[1], -y[0]], [2*y[1], 2*y[0]-3/2]])
y0=np.array([2.0, 2.0])
t0=0
tf=15
plt.figure()
plt.clf()
for h in [0.1,0.05,0.005]:
    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)

    vy=multipas(f,df,y0,vt)
    plt.plot(vy[:,0],vy[:,1],label='h={:2.3f}'.format(h))

sol_exacte=odeint(f,y0,vt)
plt.plot(sol_exacte[:,0],sol_exacte[:,1],label='sol exacte')
plt.legend(loc='best')
