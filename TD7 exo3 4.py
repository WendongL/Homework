import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

F=lambda x:np.array([x[1]*log(x[1])-x[0]*log(x[0])-3, x[0]**4+x[0]*x[1]+x[1]**3-100])
DF=lambda x: np.array([[-log(x[0])-1, log(x[1])+1], [4*x[0]**3+x[1], 3*x[1]**2+x[0]]])
CI=np.array([2.0,3.0])
sol= sp.optimize.root(F,CI)
print(sol.x)

i=0
r=1
X=[CI]
x=CI
while r>=10**(-10):
    i=i+1
    xp=x-solve(DF(x),F(x))
    r=sp.linalg.norm(xp-x, np.inf)
    x=xp

    print('n ={:4d} rel = {:25.20f} '.format(i,r))

# EXO 4

def euler_imp(f,df,y0,vt):
    y=np.asanyarray(y0).ravel()

    vy=[y]

    dim=np.size(y0)
    h=vt[1]-vt[0]

    for t in vt[1:]:
        r=1
        i=0


        z=y

        while r>=10**(-12) and i<10:
            i=i+1
            DF=np.eye(dim)-h*df(z,t)
            F=z-y-h*f(z,t)
            zp=z-solve(DF,F)
            r=sp.linalg.norm(zp-z, np.inf)
            z=zp
            if t <vt[4]:
                print('{:3d} {:20.15f}'.format(i,r))
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
    vt=np.linspace(t0,tf,int((tf-t0)/h))

    sol_imp= euler_imp(f,df,y0,vt)
    plt.subplot(1,3,i+1)
    plt.plot(sol_imp[:,0],sol_imp[:,1],label='Euler implicite')

    vt=np.linspace(t0,tf,int((tf-t0)/0.005))
    sol_exacte=odeint(f,y0,vt)
    plt.plot(sol_exacte[:,0],sol_exacte[:,1],label='Solution exacte')
    plt.legend(loc='best')

#(3)
plt.figure()
plt.clf()
f=lambda y,t: -1*np.sin(t)*y**2+2*np.tan(t)*y
df=lambda y,t: -2*np.sin(t)*y+2*np.tan(t)
y0=2/np.sqrt(3)
t0=np.pi/6
tf=np.pi/3
nuage=[]
H=2.0**-np.arange(3,12) *np.pi/3
for h in H:
    vt=np.linspace(t0,tf,int((tf-t0)/h+1))
    sol_imp=euler_imp(f,df,y0,vt)
    sol_exacte=odeint(f,y0,vt)
    err=sp.linalg.norm(sol_exacte-sol_imp, np.inf)
    nuage.append([h,err])
nuage=np.array(nuage)

plt.loglog(nuage[:,0],nuage[:,1],'o')
P=np.polyfit(np.log10(nuage[:,0]),np.log10(nuage[:,1]),1)
plt.loglog(nuage[:,0],10**P[1]*nuage[:,0]**P[0])
plt.legend(['ordre={:.4f}'.format(P[0])])