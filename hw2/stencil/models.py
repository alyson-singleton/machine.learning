#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Linear Regression Regressor
   and the Logistic Regression classifier

   Brown CS142, Spring 2019
'''
import random
import numpy as np
import matplotlib.pyplot as plt

# to test l2_loss function
true_y = np.array([1, 2, 3])
predict_baby = np.array([3, 5, 9])

def l2_loss(predictions, Y):
    '''
    Computes L2 loss (sum squared loss) between true values, Y, and predictions.

    @params:
        Y: A 1D Numpy array with real values (float64)
        predictions: A 1D Numpy array of the same size of Y
    @return:
        L2 loss using predictions for Y.
    '''
    loss = (np.sum((Y - np.transpose(predictions)) ** 2))
    return(loss)


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using either
    stochastic gradient descent or matrix inversion.
    '''
    def __init__(self, n_features, sgd=False):
        '''
        @attrs:
            n_features: the number of features in the regression problem
            sgd: Boolean representing whether to use stochastic gradient descent
            alpha: The learning rate used in SGD
            weights: The weights of the linear regression model.
        '''
        self.n_features = n_features + 1  # An extra feature added for the bias value
        self.sgd = sgd
        self.alpha = 0.005  # Tune this parameter
        self.weights = np.zeros(n_features + 1)

    def train(self, X, Y):
        '''
        Trains the LinearRegression model weights using either
        stochastic gradient descent or matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        if self.sgd:
            self.train_sgd(X, Y)
        else:
            self.train_solver(X, Y)

    def train_sgd(self, X, Y):
        '''
        Trains the LinearRegression model weights until convergence
        using stochastic gradient descent.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None. You can change this to return whatever you want, e.g. an array of loss
            values, to produce data for your project report.
        '''
        #for loss graph
#        loss_list = []
        #how to know how to stop when converges
        max = 100000
        while max > 0.05:
            #to shuffle training list
            zippedlist = list(zip(X, Y))
            random.shuffle(zippedlist)
            X, Y = zip(*zippedlist)
            #to fix weird array thing
            X = np.array(X)
            Y = np.array(Y)
            wold = self.weights
            #to calculate new weights
            for i in range(len(X)):
                gradient = np.dot((np.dot(X[i],self.weights) - Y[i]), X[i])
                wnew = self.weights - (self.alpha * gradient)
                self.weights = wnew
            #for loss graph
#            loss_list.append(l2_loss(np.dot(X[i],self.weights), Y))
            #to calculate convergence
            difference = np.absolute(wold - self.weights)
            max = difference.max()
#        print(loss_list)

    def train_solver(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            None
        '''
        self.weights = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
        #for loss graph
#        loss_list = []
#        loss_list.append(l2_loss(np.dot(X,self.weights), Y))
#        print(loss_list)
        pass

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        predictors = []
        for i in X:
            predictors.append(np.dot(i,self.weights))
        return(np.array(predictors))

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding values for each example
        @return:
            A float number which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


# to print graphs
#loss_listy = loss_list
#ys = range(len(loss_listy))
#plt.scatter(xs, ys)
#plt.title('Made by: d85eba7e', fontsize=16)
#plt.ylabel('Loss')
#plt.xlabel('Interation')
#plt.show()
#plt.savefig("path.png")

class LogisticRegression:
    '''
    Multinomial Linear Regression that learns weights by minimizing
    mean squared error using stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_features + 1, n_classes))  # An extra row added for the bias
        self.alpha = 0.2  # tune this parameter

    def train(self, X, Y):
        '''
        Trains the model, using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None. You can change this to return whatever you want, e.g. an array of loss
            values, to produce data for your project report.
        '''
        # how to know how to stop when converges
        max = 100000
        while max > 2.9:
            #to shuffle training list
            zippedlist = list(zip(X, Y))
            random.shuffle(zippedlist)
            X, Y = zip(*zippedlist)
            #to fix weird array thing
            X = np.array(X)
            Y = np.array(Y)
            wold = self.weights
            #to calculate new weights
            for i in range(len(X)):
                l = np.dot(X[i],self.weights)
                p = softmax(l)
                gradientpj = np.zeros((10,1))
                for j in range(self.n_classes):
                    if Y[i] == j:
                        gradientpj[j] = p[j] - 1
                    else: gradientpj[j] = p[j]
                gradientlw = np.outer(X[i], gradientpj)
                wnew = self.weights - (self.alpha * gradientlw)
                self.weights = wnew
            #to calculate convergence
            difference = np.absolute(wold - self.weights)
            max = difference.max()

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        labels = np.zeros(len(X))
        for i in range(len(X)):
            predictions = np.dot(X[i],self.weights)
            probabilities = softmax(predictions)
            labels[i] = np.argmax(probabilities)
        return(labels)

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        return(np.mean(self.predict(X) == Y))
