import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp

plt.ion()
plt.show()

def trapeze(f,y0,vt):
    vy=[y0]
    y=y0
    h=vt[1]-vt[0]
    for t in vt[:-1]:
        p1=f(vy[-1],t)
        p2=f(vy[-1]+h*p1,t+h)
        y=y+h/2*(p1+p2)
        vy.append(y)
    return np.array(vy)

r=1
K=1
f= lambda y,t: -np.cos(t)*y
y0=2.5
t0=0
tf=5
h=0.25
vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
solapprox=trapeze(f,y0,vt)

g=lambda y0,t:y0*np.exp(-1*np.sin(t))
vyexacte=g(y0,vt)

plt.figure(1)
plt.clf()
plt.plot(vt,solapprox,'-o')
plt.plot(vt,vyexacte)
plt.legend(['approximation','exacte'])
plt.title('exo1')
plt.figure(2)
plt.clf()
y0=0.1
t0=0
tf=2
H=[]
Err=[]

for k in range(1,11):
    h=1/2**k
    H.append(h)
    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
    solapprox=trapeze(f,y0,vt)
    vyexacte=g(y0,vt)
    e=sp.linalg.norm(abs(solapprox-vyexacte),np.inf)
    Err.append(e)
H=np.array(H)
Err=np.array(Err)

plt.loglog(H,Err,'-o')
P=np.polyfit(np.log10(H),np.log10(Err),1)
plt.loglog(H,10**P[1]*H**P[0])

plt.title('$pente={:5.3f}$'.format(P[0]))


#EXO 2 
h=0.25
plt.figure()
plt.clf()
f= lambda y,t: np.array([y[0]*(1-y[1]),y[1]*(2*y[0]-3/2)])
y0=np.array([2,2])

t0=0
tf=15

vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
solapprox=trapeze(f,y0,vt)
solexacte=odeint(f,y0,vt)

plt.plot(solapprox[:,0],solapprox[:,1],'-o')
plt.plot(solexacte[:,0],solexacte[:,1])

h=0.005
vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
solapprox=trapeze(f,y0,vt)
plt.plot(solapprox[:,0],solapprox[:,1])
    
    
plt.legend(['approximation h=0.25','exacte','approximation h=0.005'])
plt.title('$h={0}$'.format(h))

for h in [0.1,1]:
    plt.figure()
    plt.clf()
    f= lambda y,t: np.array([y[0]*(1-y[1]),y[1]*(2*y[0]-3/2)])
    y0=np.array([2,2])
    
    t0=0
    tf=15

    vt=np.linspace(t0,tf,int((tf-t0)/h)+1)
    solapprox=trapeze(f,y0,vt)
    solexacte=odeint(f,y0,vt)
    plt.plot(solapprox[:,0],solapprox[:,1],'-o')
    plt.plot(solexacte[:,0],solexacte[:,1])
    
    
    plt.legend(['approximation','exacte'])
    plt.title('$h={0}$'.format(h))
    