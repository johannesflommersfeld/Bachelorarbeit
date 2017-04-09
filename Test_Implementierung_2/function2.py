import numpy as np

#the functional that will be minimized
def E(f):
    return np.cos(np.inner(f,f)) + 0.1*np.inner(f,f)

#first order derivative of the functional
def dE(f):
    return -2*np.sin(np.inner(f,f))*f + 0.2*f

#second order derivatives of the functional
def d_md_nE(f):
    f = f[np.newaxis]
    return -2*np.sin(np.inner(f,f))*np.eye(5)-4*np.cos(np.inner(f,f))*f.T*f + 0.2*np.ones((5,5))

#time derivative of the second order derivative of the functional
def d_td_md_nE(A,f):
	dt_f = A.dot(dE(f))[np.newaxis]
	f = f[np.newaxis]
	return -4*np.cos(np.inner(f,f))*np.inner(dt_f,f)*np.eye(5) + 8*np.sin(np.inner(f,f))*np.inner(dt_f,f)*f.T*f - 4*np.cos(np.inner(f,f))*(f.T*dt_f + dt_f.T*f)
