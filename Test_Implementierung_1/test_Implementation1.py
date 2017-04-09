import numpy as np
import function1

f_start = np.array([1.5,1.5,1.5,1.5,1.5])
t0, t1, dt = 0, 1, 0.1

#r.h.s of the ode 
def G(t,f):
    A = (1-t)*np.eye(5)/f + t*function1.d_md_nE(f)
    A_inv = np.linalg.inv(A)
    return -A_inv.dot(function1.dE(f))

#integrating the ode with heun's method
def heun(f_start, t0, t1, dt):
	t = t0
	f_final = f_start
	while t < t1:
		f_intermediate = f_final + dt*G(t,f_final)
		f_final = f_final + dt/2*(G(t,f_final) + G(t + dt,f_intermediate))
 		t += dt 
	return f_final

f = f_start
while(np.linalg.norm(function1.dE(f)) > 1e-6):
	f = heun(f, t0, t1, dt)
print("f = [%g, %g, %g, %g, %g], E = %g" % (f[0], f[1], f[2], f[3], f[4],function1.E(f)))
	

