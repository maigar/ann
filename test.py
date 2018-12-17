# import ann
# import numpy as np
#
# dnet = ann.dense_net.DenseNet(2, ann.loss.SVM, [1])
# dnet.addlayer(ann.activation.Sigmoid, 2)
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y = np.array([1, 1, 0, 1])
# dnet.train(X,Y)

import ann
import numpy as np


#Softmax with Cross entropy loss
print("Softmax with Cross entropy loss:")
dnet = ann.dense_net.DenseNet(2, ann.loss.CrossEntropy, {'l_rate': 1, 'momentum': 0})
dnet.addlayer(ann.activation.Sigmoid, 2)
dnet.addlayer(ann.activation.Softmax, 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[1,0],[0,1],[0,1],[1,0]])
for i in range(1000):
    # print(dnet.train(X, Y))
    dnet.train(X, Y)

pred = dnet.predict(X)
# pred[pred > 0.5] = 1
# pred[pred <= 0.5] = 0

print("Actual classification:\n", Y)
print("Predicted classification:\n", pred)

#SVM Loss
print("\nWith SVM Loss:")
dnet2 = ann.dense_net.DenseNet(2, ann.loss.SVM, {'l_rate': 0.1, 'momentum': 0})
dnet2.addlayer(ann.activation.Sigmoid, 2)
dnet2.addlayer(ann.activation.Linear, 2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([1,0,0,1])
for i in range(1000):
    # print(dnet2.train(X, Y))
    dnet2.train(X, Y)

pred = dnet2.predict(X)
# pred[pred > 0.5] = 1
# pred[pred <= 0.5] = 0

print("Actual classification:\n", Y)
print("Predicted classification:\n", pred)



# For Linear Regression with L2 Loss
print("\nLinear Regression:")
dnet3 = ann.dense_net.DenseNet(1, ann.loss.L2Loss, {'l_rate': 0.08, 'momentum': 0})
dnet3.addlayer(ann.activation.Linear, 1)
X = np.array([1, 2, 3, 4])
Y = np.array([1, 2, 3, 4])
for i in range(1000):
	# print(dnet3.train(X, Y))
    dnet3.train(X, Y)


print("Predicted values for X = 5 is ",dnet3.predict([5])[0])
