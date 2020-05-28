#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Linear Regression Regressor

   Brown CS142, Spring 2020
'''
import random
import numpy as np


def l2_loss(predictions, Y):
    '''
    Computes L2 loss (sum squared loss) between true values, Y, and predictions.

    @params:
        Y: A 1D Numpy array with real values (float64)
        predictions: A 1D Numpy array of the same size of Y
    @return:
        L2 loss using predictions for Y.
    '''
    # TODO

    '''
    Computing the difference between the real values and the predictions,
    squaring, and then taking the sum
    '''

    difference_array = np.subtract (Y, predictions)
    return np.dot (difference_array, difference_array)

class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using either
    stochastic gradient descent or matrix inversion.
    '''
    def __init__(self, n_features):
        '''
        @attrs:
            n_features: the number of features in the regression problem
            weights: The weights of the linear regression model.
        '''
        self.n_features = n_features + 1  # An extra feature added for the bias value
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
        self.train_solver(X, Y)

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
        # TODO

        # Transposing x, and finding XtX
        transposed_x = np.transpose (X)
        left_array = np.matmul (transposed_x, X)

        # Finding XtY
        right_array = np.matmul (transposed_x, Y)

        # Inverting XtX
        left_array_inverse = np.linalg.pinv (left_array)

        # Finding (XtX)^-1*(XtY) and returning
        self.weights = np.matmul (left_array_inverse, right_array)

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        # To Do

        # Multiplying the weights by the X values to create a 1-d array, with the predicted values
        # in each row
        return np.matmul(X, self.weights)
        pass

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
