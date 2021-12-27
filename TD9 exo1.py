import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

def trapeze(f,df,y0,vt):
    y=np.asarray(y0).ravel()


    vy=[y]

    dim=np.size(y0)
    h=vt[1]-vt[0]

    for t in vt[1:]:
        r=1
        i=0


        z=y

        while r>=10**(-12) and i<10:
            i=i+1
            DF=np.eye(dim)-h/2*df(z,t)
            F=z-y-h/2*f(z,t)-h/2*f(y,t-h)
            zp=z-solve(DF,F)
            r=sp.linalg.norm(zp-z, np.inf)
            z=zp
            #if t <vt[4]:
                #print('{:3d} {:20.15f}'.format(i,r))
        y=z
        vy.append(y)
    return np.array(vy)

t0=0
tf=15

f= lambda y,t: np.array([y[0]*(1-y[1]),y[1]*(2*y[0]-3/2)])
df=lambda y,t: np.array([[1-y[1],-y[0]],[2*y[1],2*y[0]-3/2]])
y0=np.array([2,2])

plt.figure()
plt.clf()
for i,h in enumerate([0.05,0.005,0.2]):

    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)

    sol_tra= trapeze(f,df,y0,vt)
    plt.subplot(1,3,i+1)
    plt.plot(sol_tra[:,0],sol_tra[:,1],'o')

    vt=np.linspace(t0,tf,int((tf-t0)/0.005)+1)
    sol_exacte=odeint(f,y0,vt)
    plt.plot(sol_exacte[:,0],sol_exacte[:,1])

    plt.legend(['Trapeze','Solution exacte'])

plt.figure()
plt.clf()
f=lambda y,t: -1*np.sin(t)*y**2+2*np.tan(t)*y
df=lambda y,t: -2*np.sin(t)*y+2*np.tan(t)
y0=2/np.sqrt(3)
t0=np.pi/6
tf=np.pi/3
nuage=[]
H=2.0**-np.arange(3,12) *np.pi/3
g=lambda t:1/np.cos(t)
for h in H:
    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
    sol_tra=trapeze(f,df,y0,vt)
    sol_exacte=g(vt)
    err=sp.linalg.norm(sol_exacte-sol_tra[:,0], np.inf)
    nuage.append([h,err])
nuage=np.array(nuage)

plt.loglog(nuage[:,0],nuage[:,1],'o')
P=np.polyfit(np.log10(nuage[:,0]),np.log10(nuage[:,1]),1)
plt.loglog(nuage[:,0],10**P[1]*nuage[:,0]**P[0])
plt.legend(['tan={:.4f}'.format(P[0])])