#!/usr/bin/python2.7
import argparse

import network3_feq
from network3_feq import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

#define command line arguments
parser = argparse.ArgumentParser()
#parser.add_argument("-h", "--help", help="This is a small utility program to start the training process of several neural nets. You can choose between the following architectures. You can only choose one architecture, everything else will be ignored!")
parser.add_argument("-s", "--shallow", action="store_true", dest = "shallow", help="Use neural net without hidden layer.")
parser.add_argument("-ofc", "--onefullyconected", action="store_true", dest = "onefullyconected", help="Use neural net with one fully connected layer.")
parser.add_argument("-tfc", "--twofullyconected", action="store_true", dest = "twofullyconected", help="Use neural net with two fully connected layers.")
parser.add_argument("-c", "--convolutional", action="store_true", dest = "convolutional", help="Use simple convolutional net.")
parser.add_argument("-cfc", "--convolutionalfullyconnected", action="store_true", dest = "convolutionalfullyconnected", help="Use convolutional net with extra fully connected layer.")
args = parser.parse_args()

mini_batch_size = 10

#load data
training_data, validation_data, test_data = network3_feq.load_data_shared()
if(args.shallow):
    #Network without hidden Layer
    net = Network([SoftmaxLayer(n_in=784, n_out=10, p_dropout=0.5)], mini_batch_size)
elif(args.onefullyconected):
    #Network with one fully connected layer
    net = Network([FullyConnectedLayer(n_in=784, n_out=100, p_dropout=0.5), SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)], mini_batch_size)
elif(args.twofullyconected):
    #Network with two fully connected layers
    net = Network([FullyConnectedLayer(n_in=784, n_out=100, p_dropout=0.5), FullyConnectedLayer(n_in=100, n_out=100, p_dropout=0.5), SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)], mini_batch_size)
elif(args.convolutional):
    #simple ConvNet
    net = Network([
                   ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2, 2)),
                   SoftmaxLayer(n_in=20*12*12, n_out=10, p_dropout=0.5)
                   ], mini_batch_size)
elif(args.convolutionalfullyconnected):
    #ConvNet with extra fully connected layer
    net = Network([
                   ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                 filter_shape=(20, 1, 5, 5),
                                 poolsize=(2, 2)),
                   FullyConnectedLayer(n_in=20*12*12, n_out=100, p_dropout=0.5),
                   SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)
                   ], mini_batch_size)
else:
    print("No valid parameter given, a simple shallow net is used. If you would like to choose an other architecture, please specify via an proper command line argument. See --help for more details.")
    net = Network([SoftmaxLayer(n_in=784, n_out=10, p_dropout=0.5)], mini_batch_size)

net.FEM(training_data, 30, mini_batch_size, 0.2, validation_data, test_data, tmax = 0.4, lmbda = 0.1)
