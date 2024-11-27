2-Layer Neural Network
Currently has 2 layer support (WIP multilayer, some code exists but not functional)
* CMD currently supports 'help' and 'data' (refer to main.py)
* Can change iterations on training data, sample size of training data, and growth rate through CMD line (refer to main.py ARG section)
* Can change sample count on testing data, can visualize training data / predictions (refer to main.py TRAIN section)

progress_bar in main.py used as boolean toggle for progress bar

Activations has the activation functions. Must add function to 'get_layer_from_str'. Any function that is not the output function needs its derivative
