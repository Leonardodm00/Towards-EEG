import numpy as np, io, contextlib
src=open('psf_check.py').read().split('print("=== Part A')[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src)
th=np.arcsin(NA/N_OIL); T=np.tan(th)**2
for dz in (1.0,2.0,3.0):
    R=dz*np.tan(th)
    for fac in (1.05,1.2):
        r=np.linspace(0,fac*R,int(fac*R/0.005)+1)
        I=debye_I(r,dz=dz)
        rho2=np.trapezoid(r**3*I,r)/np.trapezoid(r*I,r)
        print("dz=%.1f win=%.2fR: sigma_1D=sqrt(<rho^2>/2)/dz = %.3f"%(dz,fac,np.sqrt(rho2/2)/dz))
# predictions
cos4=np.log(1+T)+1/(1+T)-1; cos4/= T/(1+T)
cos3=(2*(np.sqrt(1+T)-1)-2*(1-1/np.sqrt(1+T)))/(2*(1-1/np.sqrt(1+T)))
print("uniform %.3f  cos^4 %.3f  cos^3 %.3f"%(np.sqrt(T/4),np.sqrt(cos4/2),np.sqrt(cos3/2)))
