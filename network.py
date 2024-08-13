# Des
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # Set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # Run network over all samples
        for i in range(samples):
            # Forward pass
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

    # Train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # Training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # Forward pass
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # Backward pass
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # Calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
