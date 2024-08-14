# Description: Activation layer class for neural network.

from layer import Layer

# Activation layer inherits from abstract class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Forward pass
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Backward pass
    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error