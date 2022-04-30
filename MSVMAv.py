import numpy as np
import pickle
import os
import csv
from libmsvmav import MSVMAv


def msvmav(X_train, X_test, Y_train, Y_test, alpha, beta):
    classifier = MSVMAv(alpha, beta)
    classifier.fit(X_train, Y_train)
    return classifier.acc(X_test, Y_test)


def experiment(X, Y, indices, alpha, beta):
    result = []
    for i in range(30):
        X_train = X[indices[2][i], :].T
        X_train = np.vstack((X_train, np.ones((1, X_train.shape[1]))))
        X_test = X[indices[3][i], :].T
        X_test = np.vstack((X_test, np.ones((1, X_test.shape[1]))))
        Y_train = Y[indices[2][i]]
        Y_test = Y[indices[3][i]]
        acc = msvmav(X_train, X_test, Y_train, Y_test, alpha, beta)
        result.append(acc)
    return result


def main():
    dataset = 'australian.pkl'
    with open(dataset, 'rb') as f:
        data = pickle.load(f)
    X = data[0]
    Y = data[1]



if __name__ == '__main__':
    main()
