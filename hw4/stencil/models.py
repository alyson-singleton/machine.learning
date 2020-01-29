#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the Naive Bayes classifier

   Brown CS142, Spring 2019
"""
import random

import numpy as np


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # step 1 : estimate the class prior
            # count fraction of times each class appears
            # laplace smoothing
            # calculate probabilities

        self.priors = np.log((np.bincount(data.labels)+1) / (len(data.labels)+self.n_classes))

        # step 2 : estimate feature distributions
            # step 2a: count fraction of times each attribute appears in class for all n_classes
            # laplace smoothing
            # calculate probabilities
            # repeat for all features

        self.trained_data = np.zeros((self.n_classes,784))

        for n in range(self.n_classes):
            number_of_class_labels = np.bincount(data.labels)[n]
            classify_data = []
            for m in range(len(data.inputs)):
                if data.labels[m] == n:
                    classify_data.append(data.inputs[m])
                else: pass
            classify_data_array = np.array(classify_data)
            self.trained_data[n] = ((np.sum(classify_data_array, axis=0)+1) / (number_of_class_labels+2))


    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        # step 3 : use maximum likelihood estimation
            # flip probabilities in rows that correspond to missing features
            # take product(ie sum of log) down rows
            # multiply by priors
            # pick maximum probability and corresponding label!

        n = 784 #figure out how not to hardcode this dimension value
        probabilities = np.zeros((len(inputs),self.n_classes,n))

        for m in range(len(inputs)):
            # flip probabilities
            a = np.where(inputs[m] > 0, self.trained_data, 1-self.trained_data)
            # now we have flipped probs in trained_data for one input m
            # put all probability matrices into one large array for numpy help
            probabilities[m] = np.log(a)
        # take product(ie sum of log) down rows and add priors
        predictions = np.argmax(((np.sum(probabilities, axis=2)) + self.priors), axis=1)
        return(predictions)

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        return(np.mean(self.predict(data.inputs) == data.labels))
