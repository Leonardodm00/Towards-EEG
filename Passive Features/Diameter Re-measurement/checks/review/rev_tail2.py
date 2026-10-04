import numpy as np, io, contextlib
src=open('psf_check.py').read().split('print("=== Part A')[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src)
rg=np.arange(0,80,0.004)
y=np.linspace(0,79,40001)
for dz in (0.0,0.84):
    I=debye_I(rg,dz=dz); I=I/(2*np.pi*np.trapezoid(rg*I,rg))  # unit 2-D integral (truncated)
    out=[]
    for x in (6,10,14):
        xs=np.linspace(x-0.15,x+0.15,7)
        L=[2*np.trapezoid(np.interp(np.sqrt(xi**2+y**2),rg,I),y) for xi in xs]
        out.append(x**2*np.mean(L))
    print("dz=%.2f  x^2*LSF at x=6,10,14: %s"%(dz,np.round(out,6)))
