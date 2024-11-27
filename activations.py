import numpy as np # type: ignore

safemode = 0

def get_layer_from_str(s):
    s = s.upper()
    if(s == "RELU"):
        return 0
    if(s == "SIGMOID"):
        return 1
    if(s == "SOFTMAX"):
        return 2
    if(s == "TANH"):
        return 3

def get_activation(n, x):
    if(n == 0):
        return ReLU(x)
    
    elif(n == 1 and safemode == 1):
        return safe_sigmoid(x)
    elif(n == 1):
        return sigmoid(x)
    
    elif(n == 2 and safemode == 1):
        return safe_softmax(x)
    elif(n == 2):
        return softmax(x)
    elif(n == 3):
        return tanh(x)

def get_d_activation(n, x):
    if(n == 0):
        return d_ReLU(x)
    elif(n == 1):
        return d_sigmoid(x)
    elif(n == 2):
        return d_softmax(x)
    elif(n == 3):
        return d_tanh(x)

#exponents can cause overflow
def safe_softmax(X):
    v = X - np.max(X)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)

def safe_sigmoid(X):
    v = X - np.max(X)
    exp_nv = np.exp(-v)
    return 1 / (1 + exp_nv) 

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def ReLU(X):
    return np.maximum(X, 0)

def softmax(X):
    return np.exp(X) / sum(np.exp(X))

def tanh(X):
    return np.tanh(X)

def d_sigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))

def d_ReLU(X):
    return X > 0

def d_softmax(X):
    return X

def d_tanh(X):
    return 1 - np.tanh(X)**2