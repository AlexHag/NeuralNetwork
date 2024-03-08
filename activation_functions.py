import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_deriv(x):
    return x > 0

def LeakyReLU(x):
    return np.maximum(x, 0.01 * x)

def LeakyReLU_deriv(x):
    return np.where(x > 0, 1, 0.01)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp = np.exp(x)
    return exp / sum(exp)

def activation(name):
    if name == "relu":
        return ReLU
    elif name == "leakyrelu":
        return LeakyReLU
    elif name == "tanh":
        return tanh
    elif name == "softmax":
        return softmax
    else:
        return None

def activation_deriv(name):
    if name == "relu":
        return ReLU_deriv
    elif name == "leakyrelu":
        return LeakyReLU_deriv
    elif name == "tanh":
        return tanh_deriv
    else:
        return None