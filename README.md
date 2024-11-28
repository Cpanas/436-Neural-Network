# Multi-Layer Neural Network
Multi-Layer Neural Network that can take input images, learn off of them, then test for accuracy on interpretting them given a label.

Has tunnable amount of Layers with their respective sizes and activation functions, along with learning rate (growth), iteration count for training, ability to visualize testting, etc.

Currently has no error handling.

# How to use
Main method **main.py** is used to start. Has tunnable parameters:
> python main.py [HELP | DATA | samples = 100] iterations = 500 | learning_rate = 0.1  | test_samples = samples = 100

Other tunnable features exist at the top of main.py
* visualize = Whether or not we should use the visualize function
* plots_to_show = How many plots we should visualize
* time_per_plot = 3 Time to stay on plot visualized (0/-1 for no plots)

* dir = os.getcwd() + [subdir] = directory data is stored in

* samples = Samples to take out of training data
* test_samples = Samples to take out of testing data
* iterations = Iterations for training
* growth = Influence backpropogation has on weights/biases
* split = Ratio of data going towards training (1 - split goes towards test)

* activations = String array of activation functions, requires a minimum of two. Length is equal to amount of layers
* nodes = [-1, -1] First and last node counts are defined by the input/output space, defines amount of nodes per layer (must be size of activations)