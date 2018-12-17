#This file contains functions for activation of a layer

import numpy as np

class Linear:
    def __init__(self, w):
        # self.d = d
        # self.f = f
        self.w = w
        self.X = None

    def forward(self, inputX):
        self.X = inputX
        return np.dot(inputX, self.w)

    def backward_w(self, dz):
        return np.dot(dz.T, self.X).T

    def backward(self, dz):
        return np.dot(dz, self.w[1:].T)


class Sigmoid:
    def __init__(self):
        self.output = None
        # self.f = f
        # self.d = d
        # self.w = np.random.rand(f, d)

    def forward(self, inputX):
        self.output = 1 / (1 + np.exp(-inputX))
        return self.output

    def backward(self, dz):
        return dz * self.output * (1 - self.output)


class Softmax:
    def __init__(self):
        self.S = None
        self.output = None
        self.d = None

    def forward(self, inputX, d):
        self.S = np.exp(inputX) / np.sum(np.exp(inputX), 1)[:, None]
        # print(self.S)
        self.output = self.S[np.arange(len(self.S)), np.argmax(d, 1)][:, None]
        self.d = d
        return self.S

    def backward(self, dz):
        # print(self.output.shape, (self.d - self.S).shape)
        return dz * self.output * (self.d - self.S)


class ReLU:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
    def __init__(self):
        self.X = None
        # self.d = d

    def forward(self, inputX):
        self.X = inputX
        output = inputX.copy()
        output[output < 0] = 0
        return output

    def backward(self, dz):
        drel = self.X.copy()
        drel[drel > 0] = 1
        drel[drel <= 0] = 0
        return dz * drel

# def softmax(y, d):
#     return np.exp(y[np.arange(len(y)), d[np.arange(len(y))]]) / np.sum(np.exp(y), 1)
