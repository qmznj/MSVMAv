import numpy as np
import pickle
from libmsvmav import MSVMAv
from sklearn.model_selection import train_test_split


def msvmav(X_train, X_test, Y_train, Y_test, alpha, beta):
    classifier = MSVMAv(alpha, beta)
    classifier.fit(X_train, Y_train)
    return classifier.acc(X_test, Y_test)


def main():
    dataset = 'australian.pkl'
    with open(dataset, 'rb') as f:
        data = pickle.load(f)
    X = data[0]
    Y = data[1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Linear
    alpha = 16
    beta = 0.25
    classifier = MSVMAv(alpha, beta, 'Linear')
    classifier.fit(X_train, Y_train)
    print(classifier.acc(X_test, Y_test))

    # Gaussian Kernel
    alpha = 1e6
    beta = 1e-10
    gamma = 1e-8
    classifier = MSVMAv(alpha, beta, 'Gaussian', gamma=gamma)
    classifier.fit(X_train, Y_train)
    print(classifier.acc(X_test, Y_test))


if __name__ == '__main__':
    main()
