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

f=lambda x:(0.5-x)*np.exp(x)*(np.sin(20*x))**2
g=lambda x:x**2
a=0
b=1
Iex,err= quad(f,a,b,epsabs=10**(-14))
print(Iex)

I=[]
H=[]
for k in range(2,21):
    m=2**k
    h=(b-a)/m
    H.append(h)
    X=np.linspace(a,b,m+1)
    Y=f(X)
    Inte=np.sum((X[1:]-X[:-1])/2*(Y[1:]+Y[:-1]))
    I.append(Inte)
    

I=np.array(I)
H=np.array(H)
err=np.abs(Iex-I)
plt.loglog(H,err,'-o')
plt.loglog(H,g(H))
P=np.polyfit(np.log10(H),np.log10(err),1)
plt.loglog(H,10**P[1]*H**P[0])
plt.legend(['ligne brisée','h**2','ordre={:.4f}'.format(P[0])])


#(5)
plt.figure()
plt.clf()
g=lambda x:x**4
I=[]
H=[]
for k in range(2,13):
    m=2**k
    h=(b-a)/m
    
    X=np.linspace(a,b,m+1)
    Y=f(X)
    Y2=f((X[1:]+X[:-1])/2)
    Inte=np.sum(h/6*(Y[1:]+4*Y2+Y[:-1]))
    I.append(Inte)
    H.append(h)

I=np.array(I)
H=np.array(H)
err=np.abs(Iex-I)
plt.loglog(H,err,'-o')
plt.loglog(H,g(H))
P=np.polyfit(np.log10(H),np.log10(err),1)
plt.loglog(H,10**P[1]*H**P[0])
plt.legend(['ligne brisée','h**4','ordre={:.4f}'.format(P[0])])
