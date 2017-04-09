import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_v
net = network_v.Network([784, 30, 10])
net.SGD(training_data, 50, 10, 0.1, 1, test_data=test_data)
