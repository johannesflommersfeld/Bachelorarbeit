#### Libraries
import numpy as np
import time
import datetime
import os
#### Modules
import spinglass

t0, t1, dt = 0, 1., 0.001
hx, hz = 0.05, 0.1
dim_lattice = 5
tol = 1

#r.h.s of the ode 
def df(sg,f,t):
	A = (1-t)*np.eye(sg.dim**2)/f + t*sg.d_md_nE(f)
	A_inv = np.linalg.inv(A)
   	return -A_inv.dot(sg.dE(f))

#integrating the ode with heun's method
def heun(sg, f_start, t0, t1, dt):
	t = t0
	f_int = f_start
	f_final = f_start
	while abs(t - t1) > dt:
		df_final = df(sg,f_final,t)

		#predictor step and adapt step size
		f1 = f_final + dt*df_final
		f2 = f_final + dt/2.*df_final
		err = np.linalg.norm(f2 - f1)
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
        
		#make sure that f_int stays in the interval (0,1) to avoid divergencies
		for n in xrange(0, sg.dim**2):
			if(f1[n] < 1 and  f1[n] > 0):
				f_int[n] = f1[n]
			elif(f1[n] > 1):
				f_int[n] = 0.9999
			elif(f1[n] < 0):
				f_int[n] = 0.0001

		#corrector step
		df_int = df(sg,f_int,t+dt)
		f = f_final + dt/2*(df_final + df_int)
 	       #make sure that f_final stays in the interval (0,1) to avoid divergencies
		for n in xrange(0, sg.dim**2):
			if(f[n] < 1 and  f[n] > 0):
				f_final[n] = f[n]
			elif(f[n] >= 1):
				f_final[n] = 0.9999
			elif(f[n] <= 0):
				f_final[n] = 0.0001
		t += dt
	return f_final

#searching ground states of spin glasses	
results = []
i_max, n_max = 100, 5
E = np.zeros(i_max*n_max)
for i in xrange(0, i_max):
    sg = spinglass.Spinglass(dim_lattice, hx, hz)
    for n in xrange(0, n_max):
        f = np.ones(dim_lattice**2)*0.45 + 0.1*np.random.uniform(0,1,dim_lattice**2)
        count = 0
        while(np.linalg.norm(sg.dE(f)) > 0.1 and count < 20):
            f = heun(sg,f, t0, t1, dt)
            count+=1
        E[i*n_max+n] = sg.E(f)
    results.append([i+1,np.amin(E[i*n_max:(i*n_max + n_max)])])
    print("Instance {0} finished".format(i+1))
EN = [result[1]/dim_lattice**2 for result in results]

#printing the results
print("Minimal Energy of all {0} Instances: E/N = {1}".format(i_max, np.amin(EN)))
print("Mean Energy: <E>/N = {0}, standard deviation sqrt(Var(E))/N = {1}".format(np.mean(EN), np.std(EN)))
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
f = open(os.path.join("results","results_{0}".format(timestamp))+".txt","w")
f.write("Searching for ground states on a {0}x{0} lattice,\n".format(dim_lattice))
f.write("using the flow equation method with a stepsize of dt = {0}.\n".fromat(dt))
f.write("External fields are: hx = {0}, hz = {1}\n".format(hx,hz))
for result in results:
    f.write("Instance {0}: E/N = {1} \n".format(result[0], result[1]/dim_lattice**2))
f.write("Minimal Energy of all 100 Instances: E/N = {0}\n".format(np.amin(EN)))
f.write("Mean Energy: <E>/N = {0}, standard deviation sqrt(Var(E))/N = {1}\n".format(np.mean(EN), np.std(EN)))
f.close()
