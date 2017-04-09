import numpy as np
import function1

f_start = np.array([1.,1.,1.,1.,1.])
A_start = np.eye(5)*f_start
t0, t1, dt = 0, 1, 0.01

#r.h.s of the ode 
def G(A,f):
	return -A.dot(function1.dE(f))

#integrating the ode with heun's method
def heun(f_start, t0, t1, dt):
	t = t0
	f_final = f_start
	A = A_start
	global A_int
	while t < t1:
		f_int = f_final + dt*G(A,f_final)
		if(t == 0):		
			A_int = np.linalg.inv((1-dt)*np.eye(5)/f_int + dt*function1.d_md_nE(f_int))
		f_final = f_final + dt/2*(G(A,f_final) + G(A_int,f_int))
 		t += dt 
		return f_final


f = f_start
while(np.linalg.norm(function1.dE(f)) > 1e-6):
	f = heun(f, t0, t1, dt)
print("f = [%g, %g, %g, %g, %g], E = %g" % (f[0], f[1], f[2], f[3], f[4],function1.E(f)))
