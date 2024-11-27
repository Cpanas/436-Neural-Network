import numpy as np # type: ignore
import os
import sys
import util
import activations as acti
import network
import Layer

#Used to toggle the progress bar
progress_bar = True

def main():
    train_dir = os.getcwd() + "/_data/train"
    test_dir = os.getcwd() + "/_data/test"
    
    #DEFAULT VALUES!
    samples = 100
    test_samples = 100
    iterations = 500
    growth = 0.100

    if len(sys.argv) == 2:
        if sys.argv[1].upper() == "HELP":
            print("Arg 1: (optional, default {}): Sample amount per value".format(samples))
            print("Arg 2: (optional, default {}): Iterations to run".format(iterations))
            print("Arg 3 (optional, default {}): Growth rate (between 0 - 1)".format(growth))
            print("Arg 4 (optional, default {}): Sample amount for test case)".format(test_samples))
            print("Input 'DATA' for information on test/training data")
            return
        if sys.argv[1].upper() == "DATA":
            train_sub, train, test_sub, test = util.data_request(train_dir, test_dir)
            print("{} directories in Training with {} files \n{} directories in Testing with {} files".format(train_sub, train, test_sub, test))
            return

    #parse cmd args
    if len(sys.argv) > 1:
            samples = abs(int(sys.argv[1]))
    if len(sys.argv) > 2:
            iterations = abs(int(sys.argv[2]))
    if len(sys.argv) > 3:
            growth = min(abs(float(sys.argv[3])), 1)
    if len(sys.argv) > 4:
            test_samples = abs(int(sys.argv[4]))

    #####################################
    #I don't recommend messing with this unless you are feeling experimental
    activations = ["relu", "softmax"]
    #first and last nodes are defined by the input
    nodes = [-1, -1]
    #####################################

    #get indexes for activation functions
    funcs = []
    for i in range(0, len(activations)):
        funcs.append(acti.get_layer_from_str(activations[i]))

    X_train, Y_train = util.build_data(train_dir, samples)
    X_train, Y_train = util.unison_shuffled_copies(X_train, Y_train)

    X_test, Y_test = util.build_data(test_dir, test_samples)
    X_test, Y_test = util.unison_shuffled_copies(X_test, Y_test)

    nodes[0] = len(X_test[0]) #input space
    nodes[len(nodes) - 1] = len(set(Y_test)) #output space

    #this is very important
    X_train = X_train.T
    X_test = X_test.T

    print("Activation functions: {}".format(activations))
    print("[Input Layer Size: {} Label size: {} Learning rate: {} Iterations: {}]".format(nodes[0], nodes[len(nodes) - 1], growth, iterations))

    #Create params to start with (weights n biases)
    nn_layers = build_params(funcs, nodes)

    ########################################
    #TRAINING
    ########################################
    trained_layers = network.gradient_descent(X_train, Y_train, nn_layers, iterations, growth)

    ########################################
    #TESTING
    ########################################

    #used to print test accuracy
    network.test_set(X_test, Y_test, trained_layers)

    #used to print test accuracy with visuals + shows estimated number (this can take a tad longer)
    #plots_to_show = 10
    #time_per_plot = 3
    #network.visual_test_set(X_test, Y_test, trained_layers, plots_to_show, time_per_plot)

def build_params(funcs, nodes):
    layers = []
    top = len(funcs) - 1

    #input
    layer = Layer.Layer((np.random.rand(nodes[1], nodes[0]) - 0.5), (np.random.rand(nodes[1], 1) - 0.5), funcs[0])
    layers.append(layer)

    #additional layers
    for i in range(1, top):
        layer = Layer.Layer((np.random.rand(nodes[i + 1], nodes[i]) - 0.5), (np.random.rand(nodes[i + 1], 1) - 0.5), funcs[i])
        layers.append(layer)

    #output
    layer = Layer.Layer((np.random.rand(nodes[top], nodes[top]) - 0.5), (np.random.rand(nodes[top], 1) - 0.5), funcs[top])
    layers.append(layer)

    return layers

if __name__ == "__main__":
    main()
