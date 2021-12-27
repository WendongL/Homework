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

f=lambda t: np.exp(t)/np.log(t)
df=lambda t: (np.exp(t)*np.log(t)-np.exp(t)/t)/(np.log(t)**2)
a=2
b=4
Inte=[]

H=[]
Iex,err= quad(f,a,b,epsabs=10**(-14))
print(Iex)

for k in range(1,16):
    m=2**k
    h=(b-a)/m
    H.append(h)
    X=np.linspace(a,b,m+1)
    X0=X[:-1]
    X1=X[1:]
    Im=np.sum(2*f(X0)+4*f(X1)-h*df(X1))*h/6

    Inte.append(Im)
    

plt.figure(0)
plt.clf()
Inte=np.array(Inte)
H=np.array(H)
err=np.abs(Iex-Inte)
plt.loglog(H,err,'-o',label='ligne brisée')

P=np.polyfit(np.log10(H),np.log10(err),1)
plt.loglog(H,10**P[1]*H**P[0],label='ordre={:.4f}'.format(P[0]))
plt.legend(loc='best')


def enright(f,df,y0,vt):
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
            
            DF=-1*np.eye(dim)+2/3*h*df(z,t)-h**2/6*df(y+h*f(y,t-h),t).dot(df(z,t))
            F=-1*z+y+1/3*h*f(y,t-h)+2/3*h*f(z,t)-h**2/6*df(y+h*f(y,t),t).dot(f(z,t))
            zp=z-solve(DF,F)
            
            r=sp.linalg.norm(zp-z, np.inf)
            z=zp
            #if t <vt[4]:
            #    print('{:3d} {:20.15f}'.format(i,r))
        y=z
        vy.append(y)
    return np.array(vy)
    
t0=0
tf=2
y0=0.1

f= lambda y,t: y*(1-y)
df=lambda y,t: 1-2*y

plt.figure(1)
plt.clf()
E=[]
H=[]
for i,h in enumerate([2**(-k) for k in range(3,13)]):
    vt=np.linspace(t0,tf,int((tf-t0)/h+1))
    H.append(h)
    sol_appro= enright(f,df,y0,vt)
    sol_exacte=y0/(y0-np.exp(-1*vt)*(-1+y0))
    err=sp.linalg.norm(sol_exacte-sol_appro[:,0],np.inf)
    E.append(err)

E=np.array(E)
H=np.array(H)
plt.loglog(H,E,'-o',label='ligne brisée')

P=np.polyfit(np.log10(H),np.log10(E),1)
plt.loglog(H,10**P[1]*H**P[0],label='ordre={:.4f}'.format(P[0]))
plt.legend(loc='best')

plt.figure(2)
plt.clf()
a=30
theta0=1
t0=0
tf=5*a
h=0.01
f=lambda y,t: np.array([-1*y[1]+a*(1-y[1]**2)*y[0], y[0]])
df=lambda y,t: np.array([[a*(1-y[1]**2), -1-a*y[0]*2*y[1]], [1,0]])
y0=np.array([0, theta0])
vt=np.linspace(t0,tf,int((tf-t0)/h+1))
sol_exacte= odeint(f,y0,vt)
plt.plot(sol_exacte[:,1],sol_exacte[:,0],label='exacte')
sol_appro=enright(f,df,y0,vt)
plt.plot(sol_appro[:,1],sol_appro[:,0],label='appro')
plt.legend(loc='best')