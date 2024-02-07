import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_propagation(self, X, y, learning_rate):
        output = self.forward_propagation(X)
        error = output - y
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            self.backward_propagation(X, y, learning_rate)

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, X):
        return np.dot(X, self.weights) + self.biases

    def backward(self, error, learning_rate):
        delta = np.dot(error, self.weights.T)
        self.weights -= learning_rate * np.dot(X.T, error)
        self.biases -= learning_rate * np.sum(error, axis=0)
        return delta

class ConvolutionalLayer:
    def __init__(self, input_shape, num_filters, filter_size):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.weights = np.random.randn(num_filters, filter_size, filter_size)
        self.biases = np.zeros(num_filters)

    def forward(self, X):
        self.input = X
        batch_size, input_channels, input_height, input_width = X.shape
        output_height = input_height - self.filter_size + 1
        output_width = input_width - self.filter_size + 1
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                output[:, :, i, j] = np.sum(X[:, :, i:i+self.filter_size, j:j+self.filter_size] * self.weights, axis=(2, 3))
        return output + self.biases

    def backward(self, error, learning_rate):
        batch_size, num_filters, output_height, output_width = error.shape
        delta = np.zeros_like(self.input)
        for i in range(output_height):
            for j in range(output_width):
                delta[:, :, i:i+self.filter_size, j:j+self.filter_size] += np.sum(error[:, :, i:i+1, j:j+1] * self.weights, axis=1)
                self.weights -= learning_rate * np.sum(error[:, :, i:i+1, j:j+1] * self.input[:, :, i:i+self.filter_size, j:j+self.filter_size], axis=0)
        self.biases -= learning_rate * np.sum(error, axis=(0, 2, 3))
        return delta

class RecurrentLayer:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = np.random.randn(input_size, hidden_size)
        self.rec_weights = np.random.randn(hidden_size, hidden_size)
        self.biases = np.zeros(hidden_size)

    def forward(self, X):
        self.input = X
        batch_size, sequence_length, _ = X.shape
        hidden_state = np.zeros((batch_size, self.hidden_size))
        self.hidden_states = [hidden_state]
        for t in range(sequence_length):
            hidden_state = np.tanh(np.dot(X[:, t, :], self.weights) + np.dot(hidden_state, self.rec_weights) + self.biases)
            self.hidden_states.append(hidden_state)
        return np.array(self.hidden_states[1:])

    def backward(self, error, learning_rate):
        batch_size, sequence_length, _ = self.input.shape
        delta = np.zeros_like(self.input)
        dweights = np.zeros_like(self.weights)
        drec_weights = np.zeros_like(self.rec_weights)
        dbiases = np.zeros_like(self.biases)
        error = error[::-1]
        hidden_error = np.zeros((batch_size, self.hidden_size))
        for t in range(sequence_length)[::-1]:
            hidden_error += error[:, t, :]
            delta[:, t, :] = hidden_error
            dweights += np.dot(self.input[:, t, :].T, hidden_error)
            drec_weights += np.dot(self.hidden_states[t].T, hidden_error)
            dbiases += np.sum(hidden_error, axis=0)
            hidden_error = hidden_error.dot(self.rec_weights.T) * (1 - self.hidden_states[t] ** 2)
        self.weights -= learning_rate * dweights
        self.rec_weights -= learning_rate * drec_weights
        self.biases -= learning_rate * dbiases
        return delta
