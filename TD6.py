import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp

plt.ion()
plt.show()

a=3
t0=0
tf=5*a
p=401

f=lambda y,t: np.array([y[1],-y[0]+a*(1-y[0]**2)*y[1]])
vt=np.linspace(t0,tf,p)
Y0=(np.random.rand(10,2)-0.5)*4
for y0 in Y0:
    sol=odeint(f,y0,vt)
    plt.plot(sol[:,0],sol[:,1])
plt.axis([-2.5,2.5,-6,6])

x=np.linspace(-2.5,2.5,16)
y=np.linspace(-6,6,16)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
N=np.sqrt(DX**2+DY**2)
plt.quiver(X,Y,DX/N,DY/N,N,angles='xy')

x=np.linspace(-2.5,2.5,501)
y=np.linspace(-6,6,501)
X,Y=np.meshgrid(x,y)
DX,DY=f([X,Y],0)
plt.contour(X,Y,DX,0,colors='r',linewidths=2)
plt.contour(X,Y,DY,0,colors='b',linewidths=2)

#(2)
a=5
y0=np.array([1,0])
t0=0
tf=5*a
h=0.01
vt=np.linspace(t0,tf,int((tf-t0)/h))
vy,out=odeint(f,y0,vt,full_output=True)

plt.figure()
plt.clf()
plt.subplot(4, 1, 1)
plt.plot(vt,vy[:,0])

plt.subplot(4, 1, 2)
plt.plot(vt,vy[:,1])

plt.subplot(4, 1, 3)
plt.plot(vt[1:],out['hu'])
plt.title('le pas utilise')

plt.subplot(4, 1, 4)
plt.plot(vt[1:],out['nqu'])
plt.title('ordre')
#(3)

a=14
y0=np.array([1,0])
t0=0
tf=5*a
h=0.01
vt=np.linspace(t0,tf,int((tf-t0)/h))
vy,out=odeint(f,y0,vt,full_output=True)

plt.figure()
plt.clf()
plt.subplot(4, 1, 1)
plt.plot(vt,vy[:,0])

plt.subplot(4, 1, 2)
plt.plot(vt,vy[:,1])

plt.subplot(4, 1, 3)
plt.plot(vt[1:],out['hu'])

plt.subplot(4, 1, 4)
plt.plot(vt[1:],out['nqu'])


##EXO2

def heun(f,y0,vt):
    vy=[y0]
    y=y0

    for n,t in enumerate(vt[:-1]):
        h_n=vt[n]-vt[n-1]
        p1=f(y,t)
        p2=f(y+h*p1,t+h*p1)
        y=y+h/2*(p1+p2)
        vy.append(y)
    return np.array(vy)


def adapt12(f,y0,t0,tf,Tol):
    h_min=10**(-7)
    h_max=(tf-t0)/2
    y=y0
    t=t0
    vt=[t]
    vy=[y]
    h=h_max
    while (tf-t)>h_min/2:
        E_alpha=h/2*np.linalg.norm(f(y+h*f(y,t),t+h)-f(y,t),np.inf)
        if np.sqrt(E_alpha/Tol)<1:
            y=y+h*f(y,t)
            t=t+h
            vt.append(t)
            vy.append(y)
            h=min(h_max,h*np.sqrt(Tol/E_alpha))
        else:
            h=max(h_min,0.95*h*np.sqrt(Tol/E_alpha))
    return np.array(vt),np.array(vy)

#(b)
Tol=0.01
a=5
t0=0
tf=5*a
vt,vy=adapt12(f,y0,t0,tf,Tol)
vt1=np.linspace(t0,tf,int((tf-t0)/Tol)+1)
sol=odeint(f,y0,vt1)

plt.figure()
plt.clf()
plt.subplot(3,1,1)
plt.plot(vt1,sol[:,0])
plt.plot(vt1,sol[:,1])
plt.subplot(3,1,2)
plt.plot(vt,vy[:,0],'-o')
plt.plot(vt,vy[:,1],'-o')
plt.subplot(3,1,3)
H=[vt[0]]
for i in range(len(vt)-1):
    H.append(vt[i+1]-vt[i])
plt.plot(vt,H)

def adapt21(f,y0,t0,tf,Tol):
    h_min=10**(-7)
    h_max=(tf-t0)/2
    y=y0
    t=t0
    vt=[t]
    vy=[y]
    h=h_max
    while (tf-t)>h_min/2:
        E_alpha=h/2*np.linalg.norm(f(y+h*f(y,t),t+h)-f(y,t),np.inf)
        if np.sqrt(E_alpha/Tol)<1:
            y=y+h/2*(f(y,t)+f(y+h*f(y,t),t+h))
            t=t+h
            vt.append(t)
            vy.append(y)
            h=min(h_max,h*np.sqrt(Tol/E_alpha))
        else:
            h=max(h_min,0.95*h*np.sqrt(Tol/E_alpha))
    return np.array(vt),np.array(vy)

Tol=0.01
a=5
t0=0
tf=5*a
vt,vy=adapt21(f,y0,t0,tf,Tol)
vt1=np.linspace(t0,tf,int((tf-t0)/Tol)+1)
sol=odeint(f,y0,vt1)

plt.figure()
plt.clf()
plt.subplot(3,1,1)
plt.plot(vt1,sol[:,0])
plt.plot(vt1,sol[:,1])
plt.subplot(3,1,2)
plt.plot(vt,vy[:,0],'-o')
plt.plot(vt,vy[:,1],'-o')
plt.subplot(3,1,3)
H=[vt[0]]
for i in range(len(vt)-1):
    H.append(vt[i+1]-vt[i])
plt.plot(vt,H)

Tol=0.01
a=14
t0=0
tf=5*a
vt,vy=adapt12(f,y0,t0,tf,Tol)
vt1=np.linspace(t0,tf,int((tf-t0)/Tol)+1)
sol=odeint(f,y0,vt1)

plt.figure()
plt.clf()
plt.subplot(3,1,1)
plt.plot(vt1,sol[:,0])
plt.plot(vt1,sol[:,1])
plt.subplot(3,1,2)
plt.plot(vt,vy[:,0],'-o')
plt.plot(vt,vy[:,1],'-o')
plt.subplot(3,1,3)
H=[vt[0]]
for i in range(len(vt)-1):
    H.append(vt[i+1]-vt[i])
plt.plot(vt,H)

Tol=0.01
a=14
t0=0
tf=5*a
vt,vy=adapt21(f,y0,t0,tf,Tol)
vt1=np.linspace(t0,tf,int((tf-t0)/Tol)+1)
sol=odeint(f,y0,vt1)

plt.figure()
plt.clf()
plt.subplot(3,1,1)
plt.plot(vt1,sol[:,0])
plt.plot(vt1,sol[:,1])
plt.subplot(3,1,2)
plt.plot(vt,vy[:,0],'-o')
plt.plot(vt,vy[:,1],'-o')
plt.subplot(3,1,3)
H=[vt[0]]
for i in range(len(vt)-1):
    H.append(vt[i+1]-vt[i])
plt.plot(vt,H)