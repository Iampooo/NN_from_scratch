# Description: This file contains the abstract base layer class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # Forward pass
    def forward(self, input):
        raise NotImplementedError
    # Backward pass
    def backward(self, output_error, learning_rate):
        raise NotImplementedError
