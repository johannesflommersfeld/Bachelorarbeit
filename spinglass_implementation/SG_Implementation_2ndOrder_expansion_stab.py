#### Libraries
import numpy as np
#### Modules
import spinglass

t0, t1, dt = 0, 1., 0.001
hx, hz = 0.05, 0.1
dim_lattice = 5
tol = 1e-3

#r.h.s of the ode 
def df(A,f):
	return -A.dot(sg.dE(f))

#creates a matrix for the expansion of A
def create_B(sg,A,f,t):
	return -np.eye(sg.dim**2)*f - (1-t)*np.eye(sg.dim**2)*df(A,f) + sg.d_md_nE(f)*f*f + f*f*sg.d_td_md_nE(f, A)*t

#integrating the ode with heun's method
def heun(sg,f_start, t0, t1, dt):
	t = t0
	f_final = f_start
	f_int = f_start
	A = np.eye(sg.dim**2)*f_start
	global A_int
	global B
	while t < 1:
		df_final = df(A,f_final)
		#predictor step and adapt step size
		f1 = f_final + dt*df_final
		f2 = f_final + dt/2.*df_final
		err = np.linalg.norm(f2 - f1, np.inf)
		if err > tol:
			while err > tol/2:
				dt /= 2. 
				err /= 4.
		elif err < tol/64:
			dt = dt*4	

		if dt < 0.000001:
			dt = 0.000001
		elif dt > 0.1:
			dt = 0.1

		n=0
		while(n < sg.dim**2):
			if(f1[n] < 1 and  f1[n] > 0):
				f_int[n] = f1[n]
			elif(f1[n] > 1):
				f_int[n] = 0.9999
			elif(f1[n] < 0):
				f_int[n] = 0.0001
			n +=1

		if(t == 0):		
			A_int = np.linalg.inv((1-dt)*np.eye(sg.dim**2)/f_int + dt*sg.d_md_nE(f_int))
		else:
			B_int = create_B(sg,A_int,f_int,t + dt)
			A_int = A_int - ((dt*A_int*B_int*A_int)/f_int)/f_int
		B = create_B(sg,A,f_final,t)
		A = A - ((dt*(A*B)*A)/f_final)/f_final
		df_int = df(A_int,f_int)
		n=0
		while(n < sg.dim**2):
			f = f_final[n] + dt/2*(df_final[n] + df_int[n])
			if(f < 1 and  f > 0):
				f_final[n] = f
			elif(f > 1):
				f_final[n] = 0.9999
			elif(f < 0):
				f_final[n] = 0.0001
			n +=1
 		t += dt 
		print("t = %g, E = %g" % (t, sg.E(f_final)))
	return f_final

#searching ground states of spin glasses
sg = spinglass.Spinglass(dim_lattice, hx, hz)
f = np.random.uniform(0.6,0.9,dim_lattice**2)
finish = 1	
#while(finish > 1e-9):
f = heun(sg,f, t0, t1, dt)
finish = 0
for f_i in f:
	if (f_i < 0.9999 and f_i > 0.0001):
		i, = np.where(f == f_i)
		finish += sg.dE(f)[i] 

print("t = 1, E/N = %g" % (sg.E(f)/dim_lattice**2))
print(f)
