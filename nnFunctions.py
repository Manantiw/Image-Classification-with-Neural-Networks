import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    z1 = 1/(1+np.exp(-z))
    return z1


def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # # forward pass # #
    # adding bias
    arr = np.ones(len(train_data))
    train_data = np.hstack((train_data, np.atleast_2d(arr).T))

    # equation 1
    a = np.matmul(train_data, np.transpose(W1))

    # equation 2
    z = sigmoid(a)

    # adding bias
    arr2 = np.ones(len(z))
    z = np.hstack((z, np.atleast_2d(arr2).T))

    # equation 3
    b = np.matmul(z, np.transpose(W2))

    # equation 4
    o = sigmoid(b)

    # # back propagation # #
    n = train_data.shape[0]
    k = W2.shape[0]

    # get 1-k y matrix
    y = np.zeros((n, k))
    int_array = train_label.astype(int)
    y[np.arange(int_array.size), int_array] = 1

    # equation 7

    # getting two intermediate values
    inter0 = y * np.log(o)

    one_y = 1-y
    one_o = 1-o
    inter1 = one_y * np.log(one_o)

    # adding the intermediates and calling sum()
    inter_sum = inter0+inter1
    sum1 = np.sum(inter_sum)
    sum1 *= (-1.0) / n

    # equation 15

    # w1 sum
    w1t = np.square(W1)
    i0 = w1t.sum()

    # w2 sum
    w2t = np.square(W2)
    i0 += w2t.sum()

    # mul with lambda/2n
    i0 = (i0 * lambdaval)
    i0 = i0 / 2.0
    i0 = i0 / n

    # adding with equation 7
    obj_val = sum1 + i0

    # obj grad

    # getting grad_w2
    delta = np.subtract(o, y)
    grad_W2 = np.matmul(np.transpose(delta), z)

    # getting grad_w1
    new_z_one = 1-z
    i1 = new_z_one * z

    delta_w1 = np.matmul(delta, W2)
    i2 = i1 * delta_w1

    grad_W1 = np.matmul(np.transpose(i2), train_data)
    grad_W1 = np.delete(grad_W1, (grad_W1.shape[0]-1), axis=0)

    # finishing up grad values eq 16,17
    lambdaval_w1 = lambdaval * W1
    lambdaval_w2 = lambdaval * W2

    grad_W1 = grad_W1 + lambdaval_w1
    grad_W2 = grad_W2 + lambdaval_w2

    grad_W1 = grad_W1/n
    grad_W2 = grad_W2/n

    # creating obj_val
    grad_W1 = grad_W1.flatten()
    grad_W2 = grad_W2.flatten()

    obj_grad = np.concatenate((grad_W1, grad_W2))

    return obj_val, obj_grad


def nnPredict(W1, W2, data):
    # adding bias
    arr = np.ones(len(data))
    data = np.hstack((data, np.atleast_2d(arr).T))

    # equation 1
    a = np.matmul(data, np.transpose(W1))

    # equation 2
    z = sigmoid(a)

    # adding bias
    arr2 = np.ones(len(z))
    z = np.hstack((z, np.atleast_2d(arr2).T))

    # equation 3
    b = np.matmul(z, np.transpose(W2))

    # equation 4
    o = sigmoid(b)

    # getting labels using argmax
    labels = np.argmax(o, axis=1)
    return labels
