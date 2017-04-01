#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

BIAS = 1


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))


def delta_weight(beta, true_label, predicted_label, x):
    """

    :param beta:
    :param true_label:
    :param predicted_label:
    :param x:
    """
    delta_w = -1 * beta * (predicted_label - true_label) * x
    return delta_w


def training_perceptron(beta, X, Y, number_of_epoch=5000):
    """

    :param beta: learning rate of perceptron
    :param X: the feature set for training
    :param Y: the target value against feature set
    """

    print('Training SetUp : \nnumber_of_epoch: ', number_of_epoch, '\nbeta:', beta)
    W = np.random.rand(1, X.shape[1])
    tracking_param_list = []  # not used in training perceptron
    for epoch in range(number_of_epoch):
        track_for_all_case_true = True
        X, Y = shuffle(X, Y)
        loss = 0
        for feature_row, true_label in zip(X, Y):
            theta = np.dot(np.array(feature_row), W.T) + BIAS
            predicted_output = 1 if theta > 0 else 0
            loss += (true_label - predicted_output) ** 2

            delta_W = [delta_weight(beta, true_label, predicted_output, x) for x in feature_row]

            tracking_param_list.append(
                [feature_row, true_label, np.around(W, decimals=1), predicted_output, theta,
                 delta_W])  # not used in tuning params

            W = np.add(W, delta_W)
            track_for_all_case_true = track_for_all_case_true and (int(predicted_output) == int(true_label))

        print('epoch : ', epoch, 'loss : ', loss)
        if track_for_all_case_true:
            print('perceptron for given dataset trained...')
            break

    print_training_details(tracking_param_list)
    return number_of_epoch


def print_training_details(tracking_param_list):
    import pandas as pd
    df = pd.DataFrame(tracking_param_list,
                      columns=['feature_row', 'true_label', 'weight vector', 'predicted_output', 'theta', 'delta_W'])

    print(df)


if __name__ == '__main__':
    pd.set_option('display.width', 200)

    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    Y = np.array([0, 1, 1, 0])
    beta = 0.1

    print('Training Data : \n', pd.DataFrame([[x, y] for x, y in zip(X, Y)], columns=['X', 'Y']))

    W = training_perceptron(beta, X, Y)
