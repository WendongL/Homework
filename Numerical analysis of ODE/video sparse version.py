import numpy as np
import pylab as plt
import scipy as sp

plt.ion()
plt.show()



def f0(x):
    y=[]
    for xi in x:
        if (xi>1 and xi<=2):
            y.append(4*(xi-1))
        elif (xi>2 and xi<=3):
            y.append(-4*(xi-3))
        else:
            y.append(0)
    return y

def expl_amont(x1,xN,t0,tf,N,f0,c,V):   #x1,xN sont le domaine de definition de x, c est le rapport de dt/dx. L'ideal est de poser c=0.999999...
    X,dx = np.linspace(x1,xN,N,retstep=True)
    U0=f0(X)
    dt=dx*c
    vt,dt = np.linspace(t0,tf,int((tf-t0)/dt)+1,retstep=True)
    A=np.eye(N)*(1-V*dt/dx) + np.eye(N, k=-1)*V*dt/dx
    U=U0
    sol=[]
    for i, t in enumerate(vt):
        U=A.dot(U)
        sol.append(U)
    sol=np.array(sol)
    return X,vt,sol


##
plt.figure(1)
plt.clf()

x1=0
xN=6
t0=0
tf=0.7
N=201
c=0.24

V=4
X,vt,sol_amont=expl_amont(x1,xN,t0,tf,N,f0,c,V)

i=0
for U in sol_amont:
    plt.clf()
    plt.axis([x1,xN,-1.0,5])
    plt.title('schéma explicite décentré amont' '\n' r'$\frac{{\Delta t}}{{\Delta x}}={:3.2f}, \quad t={:4.3f}$'.format(c,vt[i]))

    plt.plot(X,U,label='schéma explicite décentré amont')
    plt.plot(X,f0(X-V*vt[i]),label='solution exacte')
    i+=1
    plt.text(5, 4, r'$\frac{{\partial u}}{{\partial t}}+{0}\frac{{\partial u}}{{\partial x}}=0$'.format(V))
    plt.legend(loc='lower left')
    plt.pause(10**-10)
plt.legend(loc='lower left')
plt.pause(5)

##
def expl_centre(x1,xN,t0,tf,N,f0,c,V):   #x1,xN sont le domaine de definition de x, c est le rapport de dt/dx. L'ideal est de poser c=0.999999...
    X,dx = np.linspace(x1,xN,N,retstep=True)
    U0=f0(X)
    dt=dx*c
    vt,dt = np.linspace(t0,tf,int((tf-t0)/dt)+1,retstep=True)
    A=np.eye(N) + np.eye(N, k=-1)*V*dt/dx/2 - np.eye(N, k=1)*V*dt/dx/2
    U=U0
    sol=[]
    for i, t in enumerate(vt):
        U=A.dot(U)
        sol.append(U)
    sol=np.array(sol)
    return X,vt,sol


##
c=0.24
X,vt,sol_centre=expl_centre(x1,xN,t0,tf,N,f0,c,V)

i=0
for U in sol_centre:
    plt.clf()
    plt.axis([x1,xN,-1.0,5])
    plt.title('schéma explicite centré est inconditionnnellement instable' '\n' r'$\frac{{\Delta t}}{{\Delta x}}={:3.2}, \quad t={:4.3f}$'.format(c,vt[i]))

    plt.plot(X,U,label='schéma explicite centré')
    plt.plot(X,f0(X-V*vt[i]),label='solution exacte')
    plt.text(5, 4, r'$\frac{{\partial u}}{{\partial t}}+{0}\frac{{\partial u}}{{\partial x}}=0$'.format(V))
    i+=1
    plt.legend(loc='lower left')
    plt.pause(10**-10)
plt.legend(loc='lower left')
plt.pause(5)

##

def Lax_Fried(x1,xN,t0,tf,N,f0,c,V):   #x1,xN sont le domaine de definition de x, c est le rapport de dt/dx. L'ideal est de poser c=0.999999...
    X,dx = np.linspace(x1,xN,N,retstep=True)
    U0=f0(X)
    dt=dx*c
    vt,dt = np.linspace(t0,tf,int((tf-t0)/dt)+1,retstep=True)
    A=np.eye(N, k=-1)*(1+V*dt/dx)/2+np.eye(N, k=1)*(1-V*dt/dx)/2
    U=U0
    sol=[]
    for i, t in enumerate(vt):
        U=A.dot(U)
        sol.append(U)
    sol=np.array(sol)
    return X,vt,sol


##
X,vt,sol_fried=Lax_Fried(x1,xN,t0,tf,N,f0,c,V)

i=0
for i in range(sol_fried.shape[0]):
    plt.clf()
    plt.axis([x1,xN,-1.0,5])
    plt.title('schémas de Lax-Friedrich et explicite amont' '\n' r'$\frac{{\Delta t}}{{\Delta x}}={:3.2}, \quad t={:4.3f}$'.format(c,vt[i]))

    plt.plot(X,sol_amont[i],label='schéma explicite amont')
    plt.plot(X,sol_fried[i],label='schéma de Lax-Friedrich')
    plt.text(5, 4, r'$\frac{{\partial u}}{{\partial t}}+{0}\frac{{\partial u}}{{\partial x}}=0$'.format(V))
    plt.plot(X,f0(X-V*vt[i]),label='solution exacte')

    plt.legend(loc='lower left')
    plt.pause(10**-10)
plt.legend(loc='lower left')
plt.pause(5)

##
def crank_schema(x1,xN,t0,tf,N,f0,c,V):   #x1,xN sont le domaine de definition de x, c est le rapport de dt/dx. L'ideal est de poser c=0.999999...
    X,dx = np.linspace(x1,xN,N,retstep=True)
    U0=f0(X)
    dt=dx*c
    vt,dt = np.linspace(t0,tf,int((tf-t0)/dt)+1,retstep=True)
    A=np.eye(N)*(1/dt)+np.eye(N, k=1)*(V/dx/4)-np.eye(N, k=-1)*(V/dx/4)
    B=np.eye(N)*(1/dt)-np.eye(N, k=1)*(V/dx/4)+np.eye(N, k=-1)*(V/dx/4)
    U=U0
    sol=[]
    for i, t in enumerate(vt):
        V=np.linalg.solve(A,B.dot(U))
        U=V
        sol.append(U)
    sol=np.array(sol)
    return X,vt,sol

##

X,vt,sol_crank=crank_schema(x1,xN,t0,tf,N,f0,c,V)

i=0
for i in range(sol_crank.shape[0]):
    plt.clf()
    plt.axis([x1,xN,-1.0,5])
    plt.title('schéma de Crank-Nikolson et explicite amont' '\n' r'$\frac{{\Delta t}}{{\Delta x}}={:3.2}, \quad t={:4.3f}$'.format(c,vt[i]))

    plt.plot(X,sol_amont[i],label='schéma explicite amont')
    plt.plot(X,sol_crank[i],label='schéma de Crank-Nikolson')
    plt.text(5, 4, r'$\frac{{\partial u}}{{\partial t}}+{0}\frac{{\partial u}}{{\partial x}}=0$'.format(V))
    plt.plot(X,f0(X-V*vt[i]),label='solution exacte')

    plt.legend(loc='upper left')
    plt.pause(10**-10)
plt.legend(loc='upper left')
plt.pause(5)

##

plt.figure(2)
plt.clf()

x1=0
xN=6
t0=0
tf=0.7

c=0.24
V=4
def reglin(schema):
    A=[10**i for i in range(2,5)]
    H=[]
    E=[]
    for N in A:
        X,vt,sol_amont=schema(x1,xN,t0,tf,N,f0,c,V)
        sol_amont_finale=sol_amont[1]
        sol_exacte_finale=f0(X-V*vt[1])
        err=np.linalg.norm(sol_amont_finale - sol_exacte_finale, np.inf)
        print(err)
        E.append(err)
        h=X[2]-X[1]
        H.append(h)
    E=np.array(E)
    H=np.array(H)
    return E,H

E,H=reglin(expl_amont)
plt.loglog(H,E,'o')
a,b=np.polyfit(np.log10(H),np.log10(E),1)
plt.loglog(H,H**a*10**b,label='explicite amont ordre={:2.1f}'.format(a))

E,H=reglin(expl_centre)
plt.loglog(H,E,'o')
a,b=np.polyfit(np.log10(H),np.log10(E),1)
plt.loglog(H,H**a*10**b,label='explicite centré ordre={:2.1f}'.format(a))

E,H=reglin(Lax_Fried)
plt.loglog(H,E,'o')
a,b=np.polyfit(np.log10(H),np.log10(E),1)
plt.loglog(H,H**a*10**b,label='Lax-Friedrich ordre={:2.1f}'.format(a))

E,H=reglin(crank_schema)
plt.loglog(H,E,'o')
a,b=np.polyfit(np.log10(H),np.log10(E),1)
plt.loglog(H,H**a*10**b,label='Crank-Nicolson ordre={:2.1f}'.format(a))

plt.legend(loc='best')
plt.title('ordre de consistance en espace')