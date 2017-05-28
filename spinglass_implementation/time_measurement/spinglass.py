#### Libraries
import numpy as np

#defines a spinglass system in a externel transversal and longitudinal field in the EA-model
class Spinglass(object):

	def __init__(self, dim, hx, hz):
		self.dim = dim
		self.hx = hx
		self.hz = hz
		self.rj = [[-2,-2], [-1,-2], [0,-2], [1,-2], [2,-2], [-2,-1], [-1,-1], [0,-1], [1,-1], [2,-1], [-2,0], [-1,0], [0,0], [1,0], [2,0], [-2,1], [-1,1], [0,1], [1,1], [2,1], [-2,2], [-1,2], [0,2], [1,2], [2,2]]
		self.J = self.createJ()

	#Energy of the square lattice with periodic boundary conditions
	def E(self,f):
		E = 0
		for i in xrange(0, self.dim**2):
			for j in xrange(0,i):
				E += self.J[i,j]*(2*f[i]-1)*(2*f[j]-1)
			E += - self.hz*(2*f[i]-1) - 2*self.hx*np.sqrt(f[i]*(1 - f[i]))
		return E

	#first order derivative of the energy
	def dE(self,f):
		out = np.zeros(self.dim**2)
		for n in xrange(0, self.dim**2):
			for j in xrange(0,self.dim**2):
				out[n] += 2*self.J[n,j]*(2*f[j] - 1)
			out[n] += -2*self.hz - self.hx*(1-2*f[n])/np.sqrt(f[n]*(1-f[n]))
		return out

	#second order derivatives of the energy
	def d_md_nE(self,f):
		out = np.empty((self.dim**2,self.dim**2))
		for n in xrange(0, self.dim**2):
			for l in xrange(0, self.dim**2):
				if (n == l):
					out[n,l] = self.hx/(2*(f[n]*(1-f[n]))**(3./2.))
				else:
					out[n,l] = 4*self.J[n,l]
		return out

	#time derivative of the second order derivative of the energy
	def d_td_md_nE(self,f,A):
		df = A.dot(self.dE(f))
		vec = -self.hx*0.75*(np.ones(self.dim**2) - 2*f)*df/(f*(1-f))**(5./2.)
		return np.eye(self.dim**2)*vec

	#create coupling between nearest neighbours with periodic boundary conditions
	def createJ(self):
		mu, sigma = 0, 1
		min, max = - self.dim/2, self.dim/2
		J = np.zeros((self.dim**2,self.dim**2))
		for n in xrange(0,self.dim**2):
			for m in xrange(0,self.dim**2):
				if (self.rj[n][0] == self.rj[m][0] and (self.rj[n][1] == self.rj[m][1] + 1 or self.rj[n][1] == self.rj[m][1] - 1)):
					rn = np.random.normal(mu, sigma)
					J[n,m] = rn
					J[m,n] = rn
				elif(self.rj[n][0] == self.rj[m][0] and ( (self.rj[n][1] == max and self.rj[m][1] == min) or (self.rj[n][1] == min and self.rj[m][1] == max) ) ):
					rn = np.random.normal(mu, sigma)
					J[n,m] = rn
					J[m,n] = rn
				elif(self.rj[n][1] == self.rj[m][1] and ( self.rj[n][0] == self.rj[m][0] + 1 or self.rj[n][0] == self.rj[m][0] - 1)):
					rn = np.random.normal(mu, sigma)
					J[n,m] = rn
					J[m,n] = rn
				elif(self.rj[n][1] == self.rj[m][1] and ( (self.rj[n][0] == max and self.rj[m][0] == min) or (self.rj[n][0] == min and rj[m][0] == max))):
					rn = np.random.normal(mu, sigma)
					J[n,m] = rn
					J[m,n] = rn
		return J

#### Miscellaneous functions			
def DiracDelta(n,m):
	if (n == m):	
		return 1
	else:
		return 0
