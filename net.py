# Sanjay Mohan
# A module containing the structure and algorithms of a simple feedforward neural network
# This module was written with reference to Michael Nielsen's digital book
# http://neuralnetworksanddeeplearning.com/
# I made use of his algebraic descriptions of the algorithms behind neural networks, but this module was written
#  with no reference to his actual Python code.

import numpy as np
import random
import gzip
import pickle
import warnings

warnings.filterwarnings('error')  # handling occasional exponential overflow errors (fixed!)


class Network:

    def __init__(self, layoutArray, layers=None):
        """
        A simple feedforward neural network
        :param layoutArray: The topology of the network (eg [784, 10, 10] has 784 input nodes, 10 hidden nodes,
        10 output nodes, and 3 total layers)
        :param layers: the layers containing weights and biases of an already trained network
        """
        if layers is None:
            self.numLayers = len(layoutArray)  # number of layers (total) in network, including input layer
            # For each layer (index) of layoutArray, make Layer with layoutArray[i] inputs and layoutArray[i+1] outputs
            # First index of layoutArray indicates number of inputs to the network, so its corresponding
            # layer does not have weights and does not need a representative instance of Layer class;
            # self.layers[0] is therefore set as None. this representation makes backpropogation clearer
            self.layers = [Layer(layoutArray[i], layoutArray[i + 1]) for i in range(self.numLayers - 1)]
            self.layers.insert(0, None)
        else:
            self.layers = layers
            self.numLayers = len(self.layers)

    def feedforward(self, inputs):
        # Computes the output of the network given input
        # inputs must have length of size layoutArray[0]
        x = inputs
        for layer in self.layers[1:]:
            x = layer.calculate(x)
        return x

    def gradientDescent(self, training, epochs, minibatchSize, lrnRate, valiData=None):
        # The gradient descent algorithm
        if lrnRate <= 0:
            raise ValueError("Learning Rate must be positive")
        # Repeat gradient calculation for each minibatch; repeat this whole iteration for each epoch
        for epoch in range(epochs):
            random.shuffle(training)
            # Create minibatches-this is called "stochastic" gradient descent; quickens learning through approximations
            trainingLength = len(training)
            minibatches = []
            first = 0
            while first < trainingLength:
                last = first + minibatchSize
                if last > trainingLength:
                    last = trainingLength
                minibatches.append(training[first:last])
                first = last
            for minibatch in minibatches:
                mbLength = len(minibatch)
                # Calculate weight and bias gradients
                gradient_w = []
                gradient_b = []
                # Initialize all gradients as 0
                for layer in self.layers[1:]:  # input layer has no weights/biases, thus no gradients
                    gradient_w.append(np.zeros((len(layer.w), len(layer.w[0]))))
                    gradient_b.append(np.zeros((len(layer.b), 1)))
                # Find average gradient
                for input in minibatch:
                    costGradient_w, costGradient_b = self.backpropagation(input[0], input[1])
                    for l in range(self.numLayers - 1):
                        gradient_w[l] += costGradient_w[l] / mbLength
                        gradient_b[l] += costGradient_b[l] / mbLength
                # Update weights and biases - first term is normal gradient, second promotes lower magnitude w and b
                for l in range(self.numLayers - 1):
                    layer = self.layers[l+1]
                    layer.w += -1 * lrnRate * (costGradient_w[l] + 1 * layer.w / trainingLength)
                    layer.b += -1 * lrnRate * costGradient_b[l]
            accuracy = ""
            # Determine accuracy on test data at end of each epoch if test data is provided
            if valiData:
                accuracy = "Accuracy = " + str(self.evaluate(valiData)) + "%"
            print("Epoch", epoch, "complete.", accuracy)
        print("Training complete")

    def backpropagation(self, input, expected):
        # Returns np.arrays of cost gradients with respect to each weight and bias
        # Feedforward, find weighted inputs and activations for each layer
        # Weighted inputs (sum of x*w+b for a layer); each layer has "z" vector except input layer
        z = [None]
        # Activations (activation of weighted inputs, or initial inputs); each layer has "a" vector
        a = [input]
        for layer in self.layers[1:]:
            z_l = layer.w.dot(a[-1]) + layer.b
            z.append(z_l)
            a.append(activation(z_l))
        # Output error d_L from last layer
        d_L = self.costPrime(expected, a[-1]) * activationPrime(z[-1])
        # Output error d_l from each layer (backpropagate)
        d = [d_L]  # d holds output errors of each layer in backwards order (L, L-1, L-2, etc.)
        for l in range(self.numLayers - 2, 0, -1):  # from second to last layer to second layer
            layer = self.layers[l + 1]  # layer 2 is in index 1, etc.
            d_l = np.transpose(layer.w).dot(d[0]) * activationPrime(z[l])
            d.insert(0, d_l)
        # Compute gradients
        costGradient_w = []
        costGradient_b = []
        for l in range(self.numLayers - 1):
            costGradient_w.append(d[l].dot(np.transpose(a[l])))
            costGradient_b.append(d[l])
        return costGradient_w, costGradient_b

    def costFunction(self, expected, output):
        # Quadratic cost function, aka mean squared error
        return 0.5 * (np.linalg.norm(expected - output)**2)

    def costPrime(self, expected, output):
        # Derivative of cost function with respect to output
        return output - expected

    def evaluate(self, data):
        # Evaluates the accuracy of this network over param data
        accuracy = 0.0
        length = len(data)
        for x in data:
            output = self.feedforward(x[0])
            if output.max() == output[x[1]]:
                accuracy += 1 / length
        return 100 * accuracy

    def saveNetwork(self, name):
        # Saves the layers of the network to a file with given name
        file = gzip.open(name, "w")
        pickle.dump(self.layers, file)
        file.close()
        print("Network", name, "saved")


class Layer:

    def __init__(self, prevNodes, nodes):
        self.size = nodes
        # Each row contains weights for one "node" in this layer
        # Weights initialized as floats, standard deviation sqrt(1/numinputs)
        self.w = np.random.randn(nodes, prevNodes) / np.sqrt(prevNodes)
        # Biases - one per "node" in this layer; initialized in same way as weights
        self.b = np.random.randn(nodes, 1)

    def calculate(self, x):
        # Returns a vector of length self.size with results of x input to this layer
        if len(x) != len(self.w[0]):  # in case improper size of inputs are input
            raise ValueError("Incorrect size of inputs: ", len(x))
        output = activation(self.w.dot(x) + self.b, self)  # dot product and element-wise addition
        return output


def activation(x, layer=None):
    # Applies activation function to each element of x
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except RuntimeWarning:
        print("OVERFLOW ERROR")
        print("x =", x)
        print("_____")
        print("b =", layer.b)
        print("_____")
        print("w =", layer.w)


def activationPrime(x):
    # Derivative of the above activation function
    return activation(x) * (1 - activation(x))


def loadNetwork(name):
    # Loads network from file with given name
    file = gzip.open(name, "rb")
    layers = pickle.load(file, encoding="latin1")
    file.close()
    return Network(None, layers)
