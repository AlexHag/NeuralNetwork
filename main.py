import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from helpers import *

network = [
    { 
        "nodes": 50,
        "activation_function": "relu",
    },
    {
        "nodes": 40,
        "activation_function": "tanh",
    },
    {
        "nodes": 36,
        "activation_function": "leakyrelu",
    },
    {
        "nodes": 36,
        "activation_function": "softmax",
    }
]

data = pd.read_csv("./data/train.csv")
data = np.array(data)
np.random.shuffle(data)
m, n = data.shape

data_train = data[1000:m]
neural_network = NeuralNetwork(network, data_train)
neural_network.init_params()
neural_network.train_model(0.1, 500)

data_test = data[:1000]

neural_network.set_data(data_test)
for i in range(15):
    neural_network.test_model(i)

save_model(neural_network.model, "./model/model.pkl")