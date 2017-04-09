import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_v1
net = network_v1.Network([784, 30, 10])
net.SGD(training_data, 50, 10, 0.4, 1.0, test_data=test_data)
