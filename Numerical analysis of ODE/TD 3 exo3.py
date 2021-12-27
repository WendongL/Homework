import numpy as np
from scipy.integrate import odeint
import pylab as plt
import scipy as sp

plt.ion()
plt.show()

def pointequilibre(a,b,c,d):
    plt.axis('scaled')
    plt.axis([-4,4,-4,4])

    Y0=np.random.rand(10,2)*2-1
    vt=np.linspace(0,2,201)
    vtn=vt[::-1]
    f=lambda y,t: np.array([a*y[0]+b*y[1], c*y[0]+d*y[1]])
    for y0 in Y0:
        sol=odeint(f,y0,vt)
        plt.plot(sol[:,0],sol[:,1])
        soln=odeint(f,y0,vtn)
        plt.plot(soln[:,0],soln[:,1])
    x=np.linspace(-4,4,16)
    y=np.linspace(-4,4,16)
    X,Y=np.meshgrid(x,y)
    DX,DY=f([X,Y],0)
    N=np.sqrt(DX**2+DY**2)
    plt.quiver(X,Y,DX/N,DY/N,N,angles='xy')

    x=np.linspace(-4,4,501)
    y=np.linspace(-4,4,501)
    X,Y=np.meshgrid(x,y)
    DX,DY=f([X,Y],0)
    plt.contour(X,Y,DX,0,colors='r',linewidths=2)
    plt.contour(X,Y,DY,0,colors='r',linewidths=2)

    A=np.array([[a,b],[c,d]])
    D,V=sp.linalg.eig(A)
    v1=V[:,0]
    v2=V[:,1]
    if np.imag(D[0]) !=0 or np.imag(D[1]) !=0:
        plt.title('$\lambda_1={:9.1}, \lambda_2={:9.1}$'.format(D[0],D[1]))
    else:
        t=np.transpose(np.linspace(-6,6,601))
        v1=np.real(v1)
        v2=np.real(v2)
        t1,vec1=np.meshgrid(t,v1)
        t2,vec2=np.meshgrid(t,v2)
        droite1=t1*vec1
        droite2=t2*vec2
        plt.plot(droite1[0,:],droite1[1,:],label='$v1=({:7.3},{:7.3})$'.format(v1[0],v1[1]))
        plt.plot(droite2[0,:],droite2[1,:],label='$v2=({:7.3},{:7.3})$'.format(v2[0],v2[1]))
        plt.title('$\lambda_1={:9.1f}, \lambda_2={:9.1f}$'.format(D[0],D[1]))
        plt.legend()



plt.figure(1)
plt.subplot(2,2,1)
pointequilibre(1,-5,1,-1)
plt.subplot(2,2,2)
pointequilibre(6,-7,5,-13)
plt.subplot(2,2,3)
pointequilibre(1,1,0.5,2)
plt.subplot(2,2,4)
pointequilibre(-4,2,1,-5)

plt.figure(2)
plt.subplot(2,2,1)
pointequilibre(0.5,5,-2,-3)
plt.subplot(2,2,2)
pointequilibre(0.5,1.5,-1,0.6)
plt.subplot(2,2,3)
pointequilibre(0,1,-1,2)
plt.subplot(2,2,4)
pointequilibre(0,-1,1,-2)