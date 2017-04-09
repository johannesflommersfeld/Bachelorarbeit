#### Libraries
import numpy as np
#### Modules
import spinglass

t0, t1, dt = 0, 1., 0.001
hx, hz = 0.5, 0.1
dim_lattice = 5

#r.h.s of the ode 
def df(A,f):
	return -A.dot(sg.dE(f))

#integrating the ode with heun's method
def heun(sg,f_start, t0, t1, dt):
	t = t0
	f_final = f_start
	A = np.eye(sg.dim**2)*f_start
	global A_int
	while t < t1:
		df_final = df(A,f_final)
		f_int = f_final + dt*df_final
		if(np.linalg.norm(f_int,np.inf) >= 1):
			break
		if(t == 0):		
			A_int = np.linalg.inv((1-dt)*np.eye(sg.dim**2)/f_int + dt*sg.d_md_nE(f_int))
		f_final = f_final + dt/2*(df_final + df(A_int,f_int))
		if(np.linalg.norm(f_final,np.inf) >= 1):
			break
 		t += dt 
		print("t = %g, E = %g" % (t, sg.E(f_final)))
	return f_final

#searching ground states of spin glasses
sg = spinglass.Spinglass(dim_lattice, hx, hz)
f = np.random.uniform(0.6,0.9,dim_lattice**2)
#while(np.linalg.norm(sg.dE(f)) > 1e-6):
f = heun(sg, f, t0, t1, dt)
print("t = 1, E = %g , dE_max = %g" % (sg.E(f), np.amax(sg.dE(f))))
