import numpy as np
from matplotlib import pyplot as plt
from activation_functions import activation, activation_deriv
from helpers import num_to_char

class NeuralNetwork:
    def __init__(self, model, data):
        self.model = model
        self.set_data(data)

    def set_data(self, data):
        self.data = data
        
        self.data_rows, self.data_columns = self.data.shape
        
        self.input_matrix = data.T[1:self.data_columns] / 255
        self.labels = data.T[0]
        
        self.one_hot_Y = self.one_hot()
    
    def init_params(self):
        for i in range(len(self.model)):
            if i == 0:
                self.model[i]["weight"] = np.random.rand(self.model[i]["nodes"], self.data_columns - 1) - 0.5
            else:
                self.model[i]["weight"] = np.random.rand(self.model[i]["nodes"], self.model[i-1]["nodes"]) - 0.5
            
            self.model[i]["bias"] = np.random.rand(self.model[i]["nodes"], 1)

    def forward_prop(self, input_matrix):
        for i in range(len(self.model)):
            if i == 0:
                self.model[i]['layer'] = self.model[i]['weight'].dot(input_matrix) + self.model[i]['bias']
            else:
                self.model[i]['layer'] = self.model[i]['weight'].dot(self.model[i-1]['layer_activation']) + self.model[i]['bias']

            self.model[i]['layer_activation'] = activation(self.model[i]['activation_function'])( self.model[i]['layer'] )
    
    def one_hot(self):
        one_hot_Y = np.zeros((self.labels.size, self.labels.max() + 1))
        one_hot_Y[np.arange(self.labels.size), self.labels] = 1
        one_hot_Y = one_hot_Y.T
        
        return one_hot_Y
    
    def backward_propagation(self):
        for i in range(len(self.model) - 1, -1, -1):

            if i == len(self.model) - 1:
                self.model[i]['d_layer'] = self.model[i]['layer_activation'] - self.one_hot_Y
            else:
                self.model[i]['d_layer'] = self.model[i+1]['weight'].T.dot(self.model[i+1]['d_layer']) * activation_deriv(self.model[i]['activation_function'])(self.model[i]['layer'])
            
            if i == 0:
                self.model[i]['d_weight'] = 1 / self.data_rows * self.model[i]['d_layer'].dot(self.input_matrix.T)
            else:
                self.model[i]['d_weight'] = 1 / self.data_rows * self.model[i]['d_layer'].dot(self.model[i-1]['layer_activation'].T)

            self.model[i]['d_bias'] = 1 / self.data_rows * np.sum(self.model[i]['d_layer'], axis=1, keepdims=True)
    
    def update_params(self, alpha):
        for i in range(len(self.model)):
            self.model[i]['weight'] = self.model[i]['weight'] - alpha * self.model[i]['d_weight']
            self.model[i]['bias'] = self.model[i]['bias'] - alpha * self.model[i]['d_bias']

    def train_model(self, alpha, itterations):
        for i in range(itterations):
            self.forward_prop(self.input_matrix)
            self.backward_propagation()
            self.update_params(alpha)
            if i % 25 == 0:
                print("itteration: ", i)
                predictions = np.argmax(self.model[-1]['layer_activation'], 0)
                accuracy = np.sum(predictions == self.labels) / self.data_rows
                print(accuracy)
    
    def test_model(self, index):
        image = self.input_matrix[:, index, None]
        self.forward_prop(image)
        prediction = np.argmax(self.model[-1]['layer_activation'], 0)

        label = self.labels[index]
        print(f"Prediction: {num_to_char(prediction[0])}")
        print(f"Label: {num_to_char(label)}")

        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.show()
