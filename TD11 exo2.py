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
for k in range(2,14):
    m=2**k
    h=(b-a)/m
    H.append(h)
    X=np.linspace(a,b,m+1)
    Y=f((X[1:]+X[:-1])/2)
    Inte=h*np.sum(Y)
    I.append(Inte)
    
plt.figure(1)
plt.clf()

I=np.array(I)
H=np.array(H)
err=np.abs(Iex-I)
plt.loglog(H,err,'-o',label='PM')
plt.loglog(H,H**2,label='H**2')
P=np.polyfit(np.log10(H),np.log10(err),1)
plt.loglog(H,10**P[1]*H**P[0], label='RL {:7.3f}'.format(P[0]))


#(2)
#K=np.arange(3,13)
#H=1/2**K
Sh=(4*I[1:]-I[:-1])/3

err=abs(Sh-Iex)
plt.loglog(H[:-1],err,'-o',label='Romberg')
P=np.polyfit(np.log10(H[:-1]),np.log10(err),1)
plt.loglog(H[:-1],10**P[1]*H[:-1]**P[0],label='RL {:8.4f}'.format(P[0]) )
plt.legend(loc='best')
