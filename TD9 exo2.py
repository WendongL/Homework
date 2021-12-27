import numpy as np
import scipy as sp
from scipy.integrate import odeint
import pylab as plt

plt.ion()
plt.show()
a=0
b=1
M=9
ua=0
ub=0
f=lambda x:(1+np.pi**2)*np.sin(np.pi*x)
h=(b-a)/(M+1)

print(h)
x,h=np.linspace(a,b,M+2,retstep=True)
print(h)
A=1/h**2*(2*np.eye(M)-np.eye(M,k=1)-np.eye(M,k=-1))+np.eye(M)
B=np.zeros(M)
B[0]=ua/h**2
B[-1]=ub/h**2
B=B+f(x[1:-1])
u=np.zeros((M+2))
u[0]=ua
u[-1]=ub
u[1:-1]=sp.linalg.solve(A,B)
plt.figure()
plt.clf()
plt.plot(x,u,label='appro')
g=lambda x:np.sin(np.pi*x)
sol_exacte=g(x)
plt.plot(x,sol_exacte,label='exacte')
plt.legend(loc='best')

##
H=[]
err=[]
for M in [19,39,79,159,319]:
    a=0
    b=1
    ua=0
    ub=0
    f=lambda x:(1+np.pi**2)*np.sin(np.pi*x)
    h=(b-a)/(M+1)

    print(h)
    x,h=np.linspace(a,b,M+2,retstep=True)
    print(h)
    A=1/h**2*(2*np.eye(M)-np.eye(M,k=1)-np.eye(M,k=-1))+np.eye(M)
    B=np.zeros(M)
    B[0]=ua/h**2
    B[-1]=ub/h**2
    B=B+f(x[1:-1])
    u=np.zeros((M+2))
    u[0]=ua
    u[-1]=ub
    u[1:-1]=sp.linalg.solve(A,B)
    g=lambda x:np.sin(np.pi*x)
    sol_exacte=g(x)
    H.append(h)
    err.append(sp.linalg.norm(sol_exacte-u, np.inf))
H=np.array(H)
err=np.array(err)
plt.loglog(H,err,'o')
a,b=np.polyfit(np.log10(H),np.log10(err),1)
plt.loglog(H,10**b*H**a,label='regression')
plt.legend(loc='best')
plt.title('ordre={:5.4f}'.format(a))
