import numpy as np
import ann.activation
import ann.loss


class Graph:
    # Computational graph class

    def __init__(self, input_dim, loss_fn, optim_config={'l_rate': 0.01, 'momentum': 0.5}):
        self.loss_fn = loss_fn
        self.input_dim = input_dim
        self.curr_gate_inp = input_dim
        # self.w=np.random.rand(input_dim, d_dim)
        self.gates = []
        self.X = None
        self.d = None
        self.w = []
        self.dw = []
        self.l_rate = optim_config['l_rate']
        self.momentum = optim_config['momentum']
        self.wgrad = []
        self.gate_feats = []

    def addgate(self, activation, units=0):
        if activation == ann.activation.Linear:
            np.random.seed(58)
            new_w = np.random.rand(self.curr_gate_inp + 1, units)
            self.w.append(new_w)
            gate = activation(new_w)
            self.dw.append(np.nan * np.ones(new_w.shape))
            self.gate_feats.append(self.curr_gate_inp + 1)
            self.curr_gate_inp = units
            self.wgrad.append(np.zeros(new_w.shape))
        else:
            self.gate_feats.append(self.curr_gate_inp)
            gate = activation()

        self.gates.append(gate)

    def set_input(self, X, Y):
        self.X = np.array(X)
        if len(self.X.shape) == 1:
            if self.input_dim != 1:
                self.X = self.X[None,:]
        # self.X = np.c_[np.ones(len(self.X)), self.X]
        if Y is not None:
            self.d = np.array(Y)
            if len(self.d.shape) == 1:
                self.d = np.reshape(self.d, ((len(self.d)), 1))

    def forward(self):
        inp = self.X.copy()
        # out = self.X.copy()
        # i = 0
        predicted_value = None
        for gate in self.gates:
            if isinstance(gate, ann.activation.Linear):
                out = gate.forward(np.c_[np.ones(len(inp)), inp])
                predicted_value = out
            elif isinstance(gate, ann.activation.Softmax):
                out = gate.forward(inp, self.d)
                predicted_value = out
                # print(self.d.shape)
                # out = out[np.arange(len(out)), np.argmax(self.d, 1)][:, None]
            else:
                out = gate.forward(inp)
                predicted_value = out

            inp = out

        # predicted_value = np.argmax(inp,1)
        return predicted_value

    def backward(self):
        y = self.forward()

        # print(y)
        # d = self.d.reshape((len(self.d), 1))
        if isinstance(self.gates[-1], ann.activation.Sigmoid) and self.loss_fn == ann.loss.CrossEntropy:
            # print('3')
            act = self.loss_fn(True)
            loss = act.loss(y, self.d)
            dldy = act.grad(y, self.d)

        else:
            # print('33')
            act = self.loss_fn()
            loss = act.loss(y, self.d)
            dldy = act.grad(y, self.d)
        prev_grad = dldy.copy()
        i = len(self.dw) - 1
        for gate in reversed(self.gates):
            # if isinstance(gate, ann.activation.Linear):
            #     dldgate_w = gate.backward_w(prev_grad)
            dldgate = gate.backward(prev_grad)
            # if isinstance(gate,ann.activation.Softmax):
            #     print(dldgate.shape)
            # print(gate)
            # print(prev_grad)
            if isinstance(gate, ann.activation.Linear):
                self.dw[i] = gate.backward_w(prev_grad)
                i -= 1

            prev_grad = dldgate
        return loss

    def update(self):
        # print(self.w)
        i = 0
        for gate in self.gates:
            if isinstance(gate, ann.activation.Linear):
                # print((self.dw[i]).shape)
                self.wgrad[i] = - self.l_rate * self.dw[i] + self.momentum * self.wgrad[i]
                self.w[i] = self.w[i] + self.wgrad[i]
                gate.w = self.w[i]
                i += 1
                # print(self.w)
