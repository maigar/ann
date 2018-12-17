import numpy as np


class L1Loss:
    def __init__(self):
        pass

    def loss(self, y, d):
        return np.sum(np.sqrt((d - y)**2))/ len(y)

    def grad(self, y, d):
        ret = np.ones(d.shape)
        ret[(d-y) > 0] *=  -1
        return ret/len(y)


class L2Loss:
    def __init__(self):
        pass

    def loss(self, y, d):
        return 1 / 2 * np.sum((d - y) ** 2) / len(y)

    def grad(self, y, d):
        return -1 * (d - y) / len(y)


class CrossEntropy:
    def __init__(self, hasSigmoid=False):
        self.hasSigmoid = hasSigmoid

    def loss(self, y, d):
        if self.hasSigmoid:
            return -(d * np.log(y) + (1 - d) * np.log(1 - y)).sum() / len(y)
        else:
            # print((-(np.log(y) * d).sum()/len(y)).shape)
            return -(np.log(y) * d).sum() / len(y)

    def grad(self, y, d):
        if self.hasSigmoid:
            return -(d / y - (1 - d) / (1 - y)) / len(y)
        else:
            # print(y)
            return np.sum((-(d / y) / len(y)), 1)[:, None]


class SVM:
    def __init__(self, hasSigmoid = True):
        self.margin = np.array([])

    def loss(self, y, d):
        correct_class_y = y[np.arange(len(y)), d.flatten()][:,None]
        loss = np.maximum(0, y - correct_class_y + 1)
        # print(y - correct_class_y)
        loss[np.arange(len(y)), d.flatten()] = 0
        self.margin = loss
        return np.sum(loss) / len(y)

    def grad(self, y, d):
        layer_grad = self.margin
        layer_grad[layer_grad > 0] = 1

        # print(layer_grad)
        layer_grad[np.arange(len(y)), d.flatten()] = 0
        layer_grad[np.arange(len(y)), d.flatten()] = -1 * np.sum(layer_grad, axis=1)
        return layer_grad
