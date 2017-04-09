import numpy as np
import function2

f_start = np.array([1.,1.,1.,1.,1.])
A_start = np.eye(5)*f_start
t0, t1, dt = 0, 1, 0.001

#r.h.s of the ode 
def G(A,f):
	return -A.dot(function2.dE(f))

#creates a matrix for the expansion of A
def create_B(A,f,t):
	return -np.eye(5)/f - (1-t)*np.eye(5)*((G(A,f)/f)/f) + function2.d_md_nE(f) + function2.d_td_md_nE(A,f)*t

#integrating the ode with heun's method
def heun(f_start, t0, t1, dt):
	t = t0
	f_final = f_start
	A = A_start
	global A_int
	global B
	while t < t1:
		f_int = f_final + dt*G(A,f_final)
		if(t == 0):		
			A_int = np.linalg.inv((1-dt)*np.eye(5)/f_int + dt*function2.d_md_nE(f_int))
		else:
			B_int = create_B(A_int, f_int,t + dt)
			A_int = A_int - dt*A_int*B_int*A_int 
		B = create_B(A, f_final,t)
		f_final = f_final + dt/2*(G(A,f_final) + G(A_int,f_int))
		A = A - dt*(A*B)*A
 		t += dt 
		return f_final


f = f_start
while(np.linalg.norm(function2.dE(f)) > 1e-6):
	f = heun(f, t0, t1, dt)
print("f = [%g, %g, %g, %g, %g], E = %g" % (f[0], f[1], f[2], f[3], f[4],function2.E(f)))
