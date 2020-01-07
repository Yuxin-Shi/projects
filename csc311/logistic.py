""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    intercept = np.ones((data.shape[0], 1))
    x = np.append(data, intercept, axis=1)
    log_odds = np.dot(x, weights)
    y = sigmoid(log_odds)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce = -(np.dot(targets.T, np.log(y)) + np.dot((1 - targets).T, np.log(1 - y)))
    y = (y >= 0.5).astype(int)
    frac_correct = np.count_nonzero(y == targets) / len(targets)
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)
    intercept = np.ones((data.shape[0], 1))
    x = np.append(data, intercept, axis=1)
    log_odds = np.dot(x, weights)
    f = np.sum(np.log(1+np.exp(log_odds)) - np.multiply(log_odds, targets))
    df = np.dot(x.T, y - targets)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """
    lmbda = hyperparameters['weight_regularization']

    y = logistic_predict(weights, data)
    intercept = np.ones((data.shape[0], 1))
    x = np.append(data, intercept, axis=1)
    log_odds = np.dot(x, weights)
    f = np.sum(np.log(1 + np.exp(log_odds)) - np.multiply(log_odds, targets)) + \
        lmbda / 2 * np.dot(weights.T, weights)

    df = np.dot(x.T, y - targets) + lmbda * weights

    return f, df, y
