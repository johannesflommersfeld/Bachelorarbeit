#### Libraries
import numpy as np

#defines a spinglass system in a externel transversal and longitudinal field in the EA-model
class Spinglass(object):

	def __init__(self, dim, hx, hz):
		self.dim = dim
		self.hx = hx
		self.hz = hz
		self.J = self.createJ()
		
	#Energy of the square lattice with periodic boundary conditions
	def E(self,f):
		i,j,k = 0,0,0
		E = 0
		for i in xrange(0, self.dim**2):
			j = (i+1)%(self.dim**2)
			k = (i+self.dim)%(self.dim**2)
			E += self.J[i,j]*(2*f[i]-1)*(2*f[j]-1) + self.J[i,k]*(2*f[i]-1)*(2*f[k]-1) - self.hz*(2*f[i]-1) - 2*self.hx*np.sqrt(f[i]*(1 - f[i]))
		return E

	#first order derivative of the energy
	def dE(self,f):
		n = 0
		m = np.empty(4).astype(int)
		out = np.zeros(self.dim**2)
		for n in xrange(0, self.dim**2):
			m[0] = (n - 1)%(self.dim**2)
			m[1] = (n - self.dim)%(self.dim**2)
			m[2] = (n + 1)%(self.dim**2)
			m[3] = (n + self.dim)%(self.dim**2)
			for j in m:					
				out[n] += 2*self.J[n,j]*(2*f[j] - 1)
			out[n] += -2*self.hz - self.hx*(1-2*f[n])/np.sqrt(f[n]*(1-f[n]))
		return out

	#second order derivatives of the energy
	def d_md_nE(self,f):
		out = np.empty((self.dim**2,self.dim**2))
		l = 0
		for n in xrange(0, self.dim**2):
			for l in xrange(0, self.dim**2):
				out[l,n] = 4*self.J[l,n]*(DiracDelta(l,n+1) + DiracDelta(l,n-1) + DiracDelta(l,n+self.dim) + DiracDelta(l, n-self.dim)) + DiracDelta(l,n)*self.hx/(2*(f[n]*(1-f[n]))**(3./2.))
		return out

	#time derivative of the second order derivative of the energy
	def d_td_md_nE(self,f,A):
		df = A.dot(self.dE(f))
		vec = -self.hx*0.75*(np.ones(self.dim**2) - 2*f)*df/(f*(1-f))**(5./2.)
		return np.eye(self.dim**2)*vec

	#create coupling between nearest neighbours with periodic boundary conditions
	def createJ(self):
		mu, sigma = 0, 1
		J = np.random.normal(mu, sigma,(self.dim**2,self.dim**2))
		return (J + J.T)/2

#### Miscellaneous functions			
def DiracDelta(n,m):
	if (n == m):	
		return 1
	else:
		return 0
