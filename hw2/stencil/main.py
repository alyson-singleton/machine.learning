#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.

   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.

   Brown CS142, Spring 2019
"""

import numpy as np
import random
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import LinearRegression, LogisticRegression

WINE_FILE_PATH = '../data/wine.txt'
MNIST_TRAIN_INPUTS_PATH = '../data/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS_PATH = '../data/train-labels-idx1-ubyte.gz'
MNIST_TEST_INPUTS_PATH = '../data/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS_PATH = '../data/t10k-labels-idx1-ubyte.gz'
MNIST_CLASSES = 10


def import_wine(filepath, test_size=0.2):
    '''
        Helper function to import the wine dataset

        @param:
            filepath: path to wine.txt
            test_size: the fraction of the dataset set aside for testing
        @return:
            X_train: training data inputs
            Y_train: training data values
            X_test: testing data inputs
            Y_test: testing data values
    '''

    # Check if the file exists
    if not os.path.exists(filepath):
        print('The file {} does not exist'.format(filepath))
        exit()

    # Load in the dataset
    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the inputs
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def import_mnist(X_train_path, Y_train_path, X_test_path, Y_test_path):
    '''
        Helper function to import the MNIST dataset

        @param:
            X_train_path: path to mnist train images
            Y_train_path: path to mnist train labels
            X_test_path: path to mnist test images
            Y_test_path: path to mnist test labels
        @return:
            X_train: training data inputs
            Y_train: training data labels
            X_test: testing data inputs
            Y_test: testing data labels
    '''
    with open(X_train_path, 'rb') as f1, open(Y_train_path, 'rb') as f2, open(X_test_path, 'rb') as f3, open(Y_test_path, 'rb') as f4:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
        inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)
        X_train = inputs / 255  # Normalizing pixel values to between 0 and 1
        Y_train = np.frombuffer(buf2, dtype='uint8', offset=8)

        buf3 = gzip.GzipFile(fileobj=f3).read(16 + 10000 * 28 * 28)
        buf4 = gzip.GzipFile(fileobj=f4).read(8 + 10000)
        test_inputs = np.frombuffer(buf3, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        X_test = test_inputs / 255  # Normalizing pixel values to between 0 and 1
        Y_test = np.frombuffer(buf4, dtype='uint8', offset=8)

    return X_train, Y_train, X_test, Y_test


def test_linreg():
    '''
        Helper function that tests LinearRegression.

        @param:
            None
        @return:
            None
    '''

    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)

    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    #### Stochastic Gradient Descent ######
    print('---------- LINEAR REGRESSION w/ SGD ----------')
    sgd_model = LinearRegression(num_features, sgd=True)
    sgd_model.train(X_train_b, Y_train)
    print('Average Training Loss:', sgd_model.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', sgd_model.average_loss(X_test_b, Y_test))

    #### Matrix Inversion ######
    print('---- LINEAR REGRESSION w/ Matrix Inversion ---')
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)
    print('Average Training Loss:', solver_model.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', solver_model.average_loss(X_test_b, Y_test))


def test_logreg():

    X_train, Y_train, X_test, Y_test = import_mnist(MNIST_TRAIN_INPUTS_PATH, MNIST_TRAIN_LABELS_PATH, MNIST_TEST_INPUTS_PATH, MNIST_TEST_LABELS_PATH)
    num_features = X_train.shape[1]

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###
    print('--------- LOGISTIC REGRESSION w/ SGD ---------')
    model = LogisticRegression(num_features, MNIST_CLASSES)
    model.train(X_train_b, Y_train)
    print("Test Accuracy: {:.1f}%".format(model.accuracy(X_test_b, Y_test) * 100))


def main():

    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0)
    np.random.seed(0)
    test_linreg()
    test_logreg()

if __name__ == "__main__":
    main()
