# Neural Networks From Scratch

In this project, I build a neural network from scratch using Python. Neural networks are powerful machine learning models that can be used for various tasks such as classification, regression, and image recognition.

## Table of Contents
- [Introduction to Neural Networks](#introduction-to-neural-networks)
- [Building Blocks of a Neural Network](#building-blocks-of-a-neural-network)
- [Implementing a Neural Network](#implementing-a-neural-network)
- [Training and Evaluating the Neural Network](#training-and-evaluating-the-neural-network)
- [Conclusion](#conclusion)

## Introduction to Neural Networks
Neural networks are inspired by the structure and function of the human brain. They consist of interconnected nodes, called neurons, organized in layers. Each neuron receives input, performs a computation, and produces an output. By adjusting the weights and biases of the neurons, neural networks can learn to make accurate predictions.

## Building Blocks of a Neural Network
A neural network is composed of several key components, including:
- [Base Layer](layer.py): Abstract base layer class for other layers.
- [Fully-Connected Layer](fc_layer.py): Perform computations and transform the input data.
- [Activation Layer](ac_layer.py): Introduces non-linearity into the network. `tanh`, `sigmoid`, and `ReLU` activation functions are implemented in [activations](activations.py).
- [Loss function](loss.py): Measures the difference between predicted and actual outputs. *Mean squared error* was implemented.
- [Network](network.py): Adjusts the weights and biases to minimize the loss.

## Implementing a Neural Network
To build a neural network from scratch, we will use Python and its scientific computing libraries such as NumPy. We will define the structure of the network, initialize the weights and biases, implement the forward and backward propagation algorithms, and update the parameters using gradient descent.

## Training and Evaluating the Neural Network
Once the neural network is implemented, we can train it on a labeled dataset. During training, the network learns to adjust its parameters to minimize the loss. After training, we can evaluate the performance of the network on unseen data to assess its accuracy and generalization capabilities.

To run the test cases, use:
```
python train_xor.py
```

## Conclusion
Building a neural network from scratch is a great way to gain a deeper understanding of how these powerful models work. By implementing the various components and training the network, we can develop a solid foundation in neural network programming.
