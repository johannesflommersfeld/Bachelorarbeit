#### Libraries
import numpy as np
#### Modules
import spinglass

class Searcher(object):
    def __init__(self, dim_lattice = 5, hx = 0.05, hz = 0.1):
        self.t0, self.t1, self.dt, self.tol = 0, 1., 0.1, 1.
        self.hx, self.hz = hx, hz
        self.dim_lattice = dim_lattice
        self.sg = spinglass.Spinglass(dim_lattice, hx, hz)
        self.f = np.ones(dim_lattice**2)*0.45 + 0.1*np.random.uniform(0,1,dim_lattice**2)

    #r.h.s of the ode 
    def df(self,f):
        return -self.sg.dE(f)

    #integrating the ode with heun's method
    def heun(self):
        t = self.t0
        f_int = self.f
        f_final = self.f
        while abs(t - self.t1) > self.dt:
            df_final = self.df(f_final)

            #predictor step and adapt step size
            f1 = f_final + self.dt*df_final
            f2 = f_final + self.dt/2.*df_final
            err = np.linalg.norm(f2 - f1)

            if self.dt < 0.000001:
                self.dt = 0.000001
            elif self.dt > 0.1:
                self.dt = 0.1

            n=0
            while(n < self.sg.dim**2):
                if(f1[n] < 1 and  f1[n] > 0):
                    f_int[n] = f1[n]
                elif(f1[n] > 1):
                    f_int[n] = 0.9999
                elif(f1[n] < 0):
                    f_int[n] = 0.0001
                n +=1

            #corrector step
            df_int = self.df(f_int)
            n=0
            f = f_final + self.dt/2.*(df_final + df_int)
            while(n < self.sg.dim**2):
                if(f[n] < 1 and  f[n] > 0):
                    f_final[n] = f[n]
                elif(f[n] >= 1):
                    f_final[n] = 0.9999
                elif(f[n] <= 0):
                    f_final[n] = 0.0001	
                n +=1
            t += self.dt
        self.f =  f_final

    #searching ground states of spin glasses
    def search(self):
        count = 0
        while(np.linalg.norm(self.sg.dE(self.f)) > 0.1 and count < 20):
            self.heun()
            count+=1
        return self.sg.E(self.f)
