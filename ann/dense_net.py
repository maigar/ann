import numpy as np
import ann


class DenseNet:
    def __init__(self, input_dim, loss_fn, optim_config={'l_rate': 0.01, 'momentum': 0.5}):
        # optim_config =[l_rate]
        """
        Initialize the computational graph object.
        """
        self.graph = ann.computational_graph.Graph(input_dim, loss_fn, optim_config)

    def addlayer(self, activation, units):
        """
        Modify the computational graph object by adding a layer of the specified type.
        """
        if activation == ann.activation.Linear:
            self.graph.addgate(ann.activation.Linear, units)
        else:
            self.graph.addgate(ann.activation.Linear, units)
            self.graph.addgate(activation)

    def train(self, X, Y):
        """
        This train is for one iteration. It accepts a batch of input vectors.
        It is expected of the user to call this function for multiple iterations.
        """
        # X1=X[np.random.randint(0,len(X))][None,:]
        # print(X1)
        # Y1=Y[np.random.randint(0,len(X))][None,:]
        # self.graph.set_input(X1, Y1)
        self.graph.set_input(X, Y)
        loss_value = self.graph.backward()
        self.graph.update()
        return loss_value

    def predict(self, X):
        """
        Return the predicted value for all the vectors in X.
        """

        self.graph.set_input(X, None)
        y = self.graph.forward()
        predicted_value = y
        if self.graph.loss_fn == ann.loss.SVM or isinstance(self.graph.gates[-1], ann.activation.Softmax):
            # print(predicted_value)
            predicted_value = np.argmax(predicted_value,axis = 1)
        return predicted_value
