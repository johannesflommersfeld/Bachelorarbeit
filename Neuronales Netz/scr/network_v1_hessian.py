"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
	self.vs = [np.random.randn(y, x+1)
                        for x, y in zip(sizes[:-1], sizes[1:])]
	for i in xrange(0,self.num_layers-1):	
		self.vs[i] = np.hstack((weights[i],biases[i]))

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
	a = np.append(a,[[1]],axis=0)
        for v in self.vs:
            a = np.append(sigmoid(np.dot(v, a)),[[1]],axis=0)
        return np.delete(a,-1,0)

    def SGD(self, training_data, epochs, mini_batch_size, dt, t1,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, dt, t1)
            if test_data:
                print "Epoch {0}: {1} / {2}, Training data: {3}/{4}".format(
                    j, self.evaluate(test_data), n_test, self.evaluateTraining(training_data), n)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, dt, t1):
        """Update the network's weights and biases by applying
        the gradient only flow equation method to a single mini batch 
	and solving it with heun's method.
        The ``mini_batch`` is a list of tuples ``(x, y)``, ``dt``
        is the step size of heun's method and ``t1`` is the end point of the homoty flow."""
	t = 0	
	while t < t1:
        	#predictor step
        	dvC = [np.zeros(v.shape) for v in self.vs]
        	hessian_out = [[np.zeros((self.sizes[-1]+1,self.sizes[1]+1)) for j in xrange(0,self.sizes[1]+1)] for i in xrange(0,self.sizes[-1]+1)]
        	hessian_hidden = [[np.zeros((self.sizes[1]+1,self.sizes[0]+1)) for j in xrange(0,self.sizes[0]+1)] for i in xrange(0,self.sizes[1]+1)]
        	hessian_mixed = [[np.zeros((self.sizes[-1]+1,self.sizes[0]+1)) for j in xrange(0,self.sizes[0]+1)] for i in xrange(0,self.sizes[1]+1)]
                hessian = [hessian_hidden,hessian_mixed,hessian_out]
		for x, y in mini_batch:
         		dvCx,hessianx = self.backprop(x, y,self.vs)
            		dvC = [(nv+dnv)/len(mini_batch) for nv, dnv in zip(dvC, dvCx)]
            		for l in xrange(0,len(hessianx)):
                                for i in xrange(0,len(hessianx[l])):
                                        for j in xrange(0,len(hessianx[l][i])):
                                                hessian[l][i][j] += hessianx[l][i][j]
		vs_int = [v - dt*dv for v, dv in zip(self.vs, dvC)]  
                hessian_int = [hessian_hidden,hessian_mixed,hessian_out]
                #corrector step
        	dvC_int = [np.zeros(v.shape) for v in self.vs]
		for x, y in mini_batch:
           		dvCx_int, hessianx_int = self.backprop(x, y,vs_int)
            		dvC_int = [(nv+dnv)/len(mini_batch) for nv, dnv in zip(dvC_int, dvCx_int)]
            		for l in xrange(0,len(hessianx)):
                                for i in xrange(0,len(hessianx[l])):
                                        for j in xrange(0,len(hessianx[l][i])):
                                                hessian[l][i][j] += hessianx[l][i][j]
		self.vs = [v - dt/2.*(dv+dv_int) for v,dv,dv_int in zip(self.vs,dvC, dvC_int)]	
		t+=dt

    def backprop(self, x, y, vs):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_v`` is a layer-by-layer list of numpy arrays,
	similar to ``self.vs``."""
	nabla_v = [np.zeros(v.shape) for v in vs]
	hessian_out = np.array((self.sizes[-1]+1,self.sizes[-2]+1))
        hessian_out = [[np.zeros((self.sizes[-1]+1,self.sizes[1]+1)) for j in xrange(0,self.sizes[1]+1)] for i in xrange(0,self.sizes[-1]+1)]
        hessian_hidden = [[np.zeros((self.sizes[1]+1,self.sizes[0]+1)) for j in xrange(0,self.sizes[0]+1)] for i in xrange(0,self.sizes[1]+1)]
        hessian_mixed = [[np.zeros((self.sizes[-1]+1,self.sizes[0]+1)) for j in xrange(0,self.sizes[0]+1)] for i in xrange(0,self.sizes[1]+1)]
        # feedforward
        activation = np.append(x,[[1]],axis=0)
        activations = [np.append(x,[[1]],axis=0)] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for v in vs:
		z = np.dot(v, activation)
        	zs.append(z)
        	activation = np.append(sigmoid(z),[[1]],axis=0)
        	activations.append(activation)
        # backward pass
        delta_out = self.cost_derivative(np.delete(activations[-1],-1, 0), y) * \
            sigmoid_prime(zs[-1])
        nabla_v[-1] = np.dot(delta_out, activations[-2].transpose())
        
        # calculation of the hessian. Works only for one hidden layer!
        for i in xrange(0,self.sizes[-1]+1):
                for j in xrange(0,self.sizes[-2]+1):
                        for k in xrange(0,self.sizes[-2]+1):
                                hessian_out[i][j][i,k] = activations[-2][j]*activations[-2][k]*self.H(activations[-1], y)[i]

        for i in xrange(0,self.sizes[1]+1):
                for j in xrange(0,self.sizes[0]+1):
                        for k in xrange(0,self.sizes[1]+1):
                                for l in xrange(0,self.sizes[0]+1):
                                        fac = 0
                                        for m in xrange(0,self.sizes[-1]):
                                                fac += vs[-1][m,i]*vs[-1][m,k]*self.H(activations[-1], y)[m]
                                        hessian_hidden[i][j][k,l] = activations[0][j]*activations[0][l]*(activations[1][k]*(1-activations[1][k])*(1-2*activations[1][k]) * \
                                                                    np.dot(vs[-1].transpose(), delta_out)[k] + activations[1][k]*(1-activations[1][k]) * \
                                                                    activations[1][i]*(1-activations[1][i])*fac)

        for i in xrange(0,self.sizes[1]+1):
                for j in xrange(0,self.sizes[0]+1):
                        for k in xrange(0,self.sizes[-1]+1):
                                for l in xrange(0,self.sizes[1]+1):
                                        hessian_mixed[i][j][k,l] = activations[0][j]*activations[1][i]*(1-activations[1][i])*(delta_out[k]*np.eye(sizes[1]+1)[i,l] + \
                                                                   activations[1][l]*vs[-1][i,k]*self.H(activations[-1], y)[k])
        hessian = [hessian_hidden,hessian_mixed,hessian_out]

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        delta = delta_out
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(np.delete(vs[-l+1].transpose(), -1, 0), delta) * sp
            nabla_v[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_v,hessian)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluateTraining(self, training_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        training_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in training_data]
        return sum(int(x == y) for (x, y) in training_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def H(self, a ,y):       
        return a*(1-a)*(1-2*a)*self.cost_derivative(a,np.append(y,[[1]],axis=0)) + (a*(1-a))**2
    
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
