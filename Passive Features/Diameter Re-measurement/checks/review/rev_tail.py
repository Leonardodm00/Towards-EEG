import numpy as np
from scipy.special import j1
NA,LAM=1.4,0.55
def airy(r):
    v=2*np.pi*NA*r/LAM; v=np.where(v==0,1e-12,v); return (2*j1(v)/v)**2
y=np.linspace(0,400,4000001)
for x in (2,4,8,16):
    xs=np.linspace(x-0.2,x+0.2,9)   # average over ~one ring period
    L=[2*np.trapezoid(airy(np.sqrt(xi**2+y**2)),y) for xi in xs]
    print(x, "x^2*LSF=%.4e  x^2.5*LSF=%.4e"%(x**2*np.mean(L), x**2.5*np.mean(L)))
