import numpy as np
from Data import spiral_data

np.random.seed(0)

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.__weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Transpose is not needed 
        self.__biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.__output = np.dot(inputs, self.__weights) + self.__biases

        return self.__output

    def __str__(self):
        print(self.__output)

class Activation_ReLU: # Rectified Linear

    def forward(self, inputs):
        self.__output = np.maximum(0, inputs)

        return self.__output

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Protect from overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.__outputs = probabilities

        return self.__outputs

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likeliness = -np.log(correct_confidences)

        return negative_log_likeliness



if __name__ == "__main__":
    
    X, y = spiral_data(points=100, classes=3)

    d1 = Layer_Dense(2, 3) # Two features: x and y positions
    a1 = Activation_ReLU()

    d2 = Layer_Dense(3, 3)
    a2 = Activation_Softmax()

    i1 = d1.forward(X)
    aO = a1.forward(i1)

    i2 = d2.forward(aO)
    a2O = a2.forward(i2)

    print(a2O[:5])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(a2O, y)
    print("Loss:", loss)







