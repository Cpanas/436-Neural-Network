import numpy as np # type: ignore
import os
import sys
import util
import activations as acti
import network
import Layer


#hashmap for directories - allows non-numerical labels
label_hash = {}

#####################################################
#####################################################
#####################################################
#####################################################

#Used to toggle the progress bar
progress_bar = True

#Used to toggle on if it'll visualize the test data
visualize = True
plots_to_show = 5 #plots it'll visualize out of training data
time_per_plot = 3 #seconds between plot visuals (set to -1 for manual)

#DIRECTORIES
dir = os.getcwd() + "/_data/emnist" #directory for test/train data

#DEFAULT VALUES
samples = 100 #samples to take out of train_dir
test_samples = 100 #samples to take out of test_dir
iterations = 500 #iterations for training
growth = 0.100 #influence backpropogation has on weights/biases
split = 0.8 #values going towards training data

#ACTIVATION FUNCTIONS
activations = ["relu", "softmax"] #activation functions
nodes = [-1, -1] #first and last node counts are defined by the input/output space
#All values inbetween decide the neuron count for its respective layer
#i.e., nodes[1] = 24 means layer 2 has 24 neurons

#####################################################
#####################################################
#####################################################
#####################################################


def main():
    #Scope
    global nodes
    global growth
    global iterations
    global test_samples
    global samples
    global dir
    global split
    global time_per_plot
    global plots_to_show
    global visualize
    global label_hash

    if len(sys.argv) == 2:
        if sys.argv[1].upper() == "HELP":
            print("Arg 1: (optional, default {}): Sample amount per value".format(samples))
            print("Arg 2: (optional, default {}): Iterations to run".format(iterations))
            print("Arg 3 (optional, default {}): Growth rate (between 0 - 1)".format(growth))
            print("Arg 4 (optional, default {}): Sample amount for test case)".format(test_samples))
            print("Input 'DATA' for information on test/training data")
            return
        if sys.argv[1].upper() == "DATA":
            sub, count = util.data_request(dir)
            print("{} directories with {} files".format(sub, count))
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
    else:
        test_samples = samples

    #get indexes for activation functions
    funcs = []
    for i in range(0, len(activations)):
        funcs.append(acti.get_layer_from_str(activations[i]))

    X_train, Y_train = util.build_data(dir, samples, split, True)
    X_train, Y_train = util.unison_shuffled_copies(X_train, Y_train)

    X_test, Y_test = util.build_data(dir, test_samples, split, False)
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

    if visualize == True:
        #used to print test accuracy with visuals + shows estimated number (this can take a tad longer)
        network.visual_test_set(X_test, Y_test, trained_layers, plots_to_show, time_per_plot)
    else:
        #used to print test accuracy
        network.test_set(X_test, Y_test, trained_layers)

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
