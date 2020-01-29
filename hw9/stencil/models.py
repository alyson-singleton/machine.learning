import numpy as np
import random


def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y A 1D Numpy array with real values (float64)
        :param predictions A 1D Numpy array of the same size of Y
        :return L2 loss using predictions for Y.
    '''
    loss = (np.sum((Y - np.transpose(predictions)) ** 2))
    return(loss)


def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x A scalar or Numpy array
        :return Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x A scalar or Numpy array
        :return Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    derivative = np.multiply(sigmoid(x),(1 - sigmoid(x)))
    return(derivative)

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=25, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        # initialize weights
        self.weights = np.zeros(len(np.transpose(X)))
        # run for number of epochs we are given
        for i in range(epochs):
            zippedlist = list(zip(X, Y))
            random.shuffle(zippedlist)
            X, Y = zip(*zippedlist)
            #to fix weird array thing
            X = np.array(X)
            Y = np.array(Y)
            #to calculate new weights
            for i in range(len(X)):
                gradient = np.dot((np.dot(X[i],self.weights) - Y[i]), X[i])
                weights = self.weights - (learning_rate * gradient)
                self.weights = weights

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        predictors = []
        for i in X:
            predictors.append(np.dot(i,self.weights))
        return(np.array(predictors))

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]

class TwoLayerNN:

    def __init__(self, hidden_size=10, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

    def train(self, X, Y, learning_rate=0.01, epochs=30, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        #initialize weight matrices
        weights1 = np.random.normal(size=(self.hidden_size,np.size(X,1)+1))
        weights2 = np.random.normal(size=(self.hidden_size+1))

        # run for number of epochs we are given
        for i in range(epochs):
            #shuffle
            zippedlist = list(zip(X, Y))
            np.random.shuffle(zippedlist)
            X, Y = zip(*zippedlist)
            #to fix weird array thing
            X = np.array(X)
            Y = np.array(Y)

            for i in range(len(X)):
                #initialize
                vinput = np.zeros(np.size(X,1)+1)
                vhidden = np.zeros(self.hidden_size+1)
                voutput = 0
                bias = np.array([1])

                #forward pass
                vinput = np.concatenate((X[i],bias))
                vhidden = self.activation(np.concatenate((np.dot(weights1,vinput),bias)))
                voutput = np.dot(vhidden,weights2)
                #print(voutput)

                #backpropagation
                d2 = 2 * (voutput-Y[i])
                #print(d2)
                dldw1 = vhidden * d2
                #print(dldw1)

                d1 = np.multiply(np.dot(weights2,d2),self.activation_derivative(
                    np.concatenate((np.dot(weights1,vinput),bias))))
                #print(d1)
                dldw0 = np.outer(d1,vinput)
                #print(dldw0)

                #to calculate new weights
                weights1 = weights1 - (learning_rate*dldw0[:(len(dldw0) - 1)])
                weights2 = weights2 - (learning_rate*dldw1)

            self.weights = [weights1,weights2]



    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        predictions = np.zeros(np.size(X, 0))
        for i in range(len(X)):
            vinput = np.zeros(np.size(X,1)+1)
            vhidden = np.zeros(self.hidden_size+1)
            bias = np.array([1])

            vinput = np.concatenate((X[i],bias))
            vhidden = self.activation(np.concatenate((np.dot(self.weights[0],vinput),bias)))
            voutput = np.dot(vhidden,self.weights[1])

            predictions[i] = np.dot(vhidden,self.weights[1])

        return(predictions)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
