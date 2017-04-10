import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_v1_hessian
net = network_v1_hessian.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.3, 1.0, test_data=test_data)
