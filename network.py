import activations as act
import numpy as np # type: ignore
import util
import main
import matplotlib.pyplot as plt # type: ignore

def gradient_descent(X, Y, layers, iterations, alpha):
    for i in range(iterations):
        Z, A = forward(layers, X)
        dW, db = backprop(layers, Z, A, X, Y)
        layers = update_params(layers, dW, db, alpha)

        if main.progress_bar is True:
            util.progress_bar(i, iterations)

    if main.progress_bar is True:
        util.progress_bar(iterations, iterations)

    print("Trained Accuracy: ", util.get_accuracy(util.get_predictions(A[len(A) - 1]), Y))
    return layers

#Used to get accuracy on the test set
def test_set(X, Y, layers):
    Z, A = forward(layers, X)
    print("Test Accuracy: ", util.get_accuracy(util.get_predictions(A[len(A) - 1]), Y))

#Used to get accuracy on the test set AND visualize it
def visual_test_set(X, Y, layers, iterations, time):
    Z, A = forward(layers, X)
    print("Test Accuracy: ", util.get_accuracy(util.get_predictions(A[len(A) - 1]), Y))

    plt.show(block=False)
    for i in range(0, iterations):
        print("Predicted: {} Actual: {}".format(util.get_predictions(A[len(A) - 1])[i], Y[i]))
        util.rebuild_image(X.T[i], 28, time)
    plt.close()

def forward(layers, X):
    A = [] #activation step
    Z = [] #bias step
    top = len(layers) - 1

    #input
    Z.append(layers[0].weight.dot(X) + layers[0].bias)
    A.append(act.get_activation(layers[0].activation, Z[0]))

    #additional layers
    for i in range(1, top):
        Z.append(layers[i].weight.dot(A[i - 1]) + layers[i].bias)
        A.append(act.get_activation(layers[i].activation, Z[i])) 

    #output
    Z.append(layers[1].weight.dot(A[top - 1]) + layers[top].bias)
    A.append(act.get_activation(layers[top].activation, Z[top]))

    return Z, A

#get array suitable for A[top]
def one_hot(Y):
    ohY = np.zeros((Y.size, Y.max() + 1))
    ohY[np.arange(Y.size), Y] = 1
    ohY = ohY.T
    return ohY

def backprop(layers, Z, A, X, Y):
    top = len(layers) - 1
    m = Y.size

    dZ = []
    dW = []
    db = []

    #calculate loss based on output A[top]
    ohY = one_hot(Y)
    dZ.append(A[top] - ohY)
    dW.append(1 / m * dZ[0].dot(A[top - 1].T))
    db.append(1 / m * np.sum(dZ[0]))

    #additonal layers
    #TODO get this to work
    for i in range(1, top):
        dZ.append(layers[top - i + 1].weight.T.dot(dZ[i - 1]) * act.get_d_activation(layers[top - i].activation, Z[top - i]))
        dW.append(1 / m * dZ[i].dot(Z[top - i - 1].T))
        db.append(1 / m * np.sum(dZ[i]))

    #output
    dZ.append(layers[1].weight.T.dot(dZ[top - 1]) * act.get_d_activation(layers[0].activation, Z[0]))
    dW.append(1 / m * dZ[top].dot(X.T))
    db.append(1 / m * np.sum(dZ[top]))

    #backprop is done backwards - reverse for relation
    dW.reverse()
    db.reverse()
    return dW, db

#used to update the layers objects
def update_params(layers, dW, db, alpha):
    for i in range(0, len(layers)):
        layers[i].weight = layers[i].weight - alpha * dW[i]
        layers[i].bias = layers[i].bias - alpha * db[i]
    return layers





