import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp
from numpy import log
from scipy.linalg import solve

plt.ion()
plt.show()

#(3)
a=0
b=1
M=49
sigma=10
nu=0.1
ua=2.0
ub=0.2
w=lambda x:2-1.8*x
k=0.05
T=1
x,h=np.linspace(a,b,M+2,retstep=True)
vt,k=np.linspace(0,T,int(T/k)+1,retstep=True)
A=nu/h**2*(-2*np.eye(M,M)+np.eye(M,M,1)+np.eye(M,M,-1))

B=np.array([0]*(M))
B[0]=nu/h**2*ua
B[-1]=nu/h**2*ub
F=lambda Y,t: A.dot(Y)+B-sigma*Y/(1+Y)
Y0=w(x)
Y0[0]=ua
Y0[-1]=ub
sol=odeint(F,Y0[1:-1],vt)

plt.figure()
plt.clf()
for i, y in enumerate(sol):
    plt.clf()
    plt.axis([a,b,-1.0,3])
    plt.title('t={:10.5f}'.format(vt[i]))
    y=np.concatenate(([ua],y,[ub]))
    plt.plot(x,y)
    plt.pause(10**-5)

#(5)
plt.figure()
plt.clf()
A=-k*nu/h**2*(np.eye(M,k=1)+np.eye(M,k=-1))+(1+2*k*nu/h**2)*np.eye(M)
S=np.array([0]*M)
S[0]=nu/h**2*ua
S[-1]=nu/h**2*ub
U=w(x)
U[0]=ua
U[-1]-ub
for t in vt:
    plt.clf()
    plt.axis([a,b,-1.0,3])
    B=k*sigma/(1+U[1:-1])
    B=np.diag(B)
    U[1:-1]=solve(A+B,U[1:-1]+k*S)
    plt.title('t={:10.5f}'.format(t))
    plt.plot(x,U)
    plt.pause(10**-5)

#(8)
X=x[1:-1]
U=w(X)

V=U
A=-1*nu/h**2*(-2*np.eye(M)+np.eye(M,k=1)+np.eye(M,k=-1))
i=0
r=1
R=np.zeros(U.shape[0])
R[0]=nu/h**2*ua
R[-1]=nu/h**2*ub
while r>=10**(-18) and i<20:
    i=i+1
    DG=A+sigma*np.eye(M)/(1+U)**2
    G=A.dot(U)+sigma*U/(1+U)-R
    V=U-solve(DG,G)
    r=np.linalg.norm(U-V,np.inf)
    print('n ={:4d} err = {:25.20f}' .format(i,r))
    U=V

V=np.concatenate(([ua],V,[ub]))
plt.plot(x,V,'o')
plt.legend(['exo5','exo8'])

#Une question: J'avais mis U en dimension (M+2), et je calculais la boucle en gardant U[1:-1],et la même boucle qu'ici s'arrête en un seul tour. Maintenant j'enlève les bord de U et V, et ça marche. Je ne comprends pas pourquoi...