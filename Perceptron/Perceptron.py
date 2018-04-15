import random
import numpy as np

class Perceptron:
    weigths = []
    lr = None

    # Give in input n inputs neuron and a learning rate
    def __init__(self, n, lr_):
        for i in range(n):
            self.weigths.append(random.uniform(-1,1))

        self.lr = lr_

    # Return the weights
    def getWeights(self):
        print(self.weigths);


    # Predicit the output
    def predict(self, inputs):
        sum_ = 0;

        for i in range(0,len(self.weigths)):
            # inputs = point  why ?
            sum_ = sum_ + self.weigths[i] * inputs[i]

        # Calling activaion function -> np.sign
        output = np.sign(sum_)
        return output;


    # Give a input array whit targets, train the network
    def train(self, inputs, target):
        for input_ in inputs:

            guess = self.predict(inputs)
            error = target - guess;

            for w in range(0, len(self.weigths)):
                self.weigths[w] = self.weigths[w] + (error*inputs[w]*self.lr)
