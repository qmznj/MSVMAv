import numpy as np
from tqdm import tqdm


class MSVMAv:
    def __init__(self, alpha, beta, kernel='Linear', bias=True, boost=False):
        self.w = None
        self.alpha = alpha
        self.beta = beta
        self.boost = boost

    def __initialize(self, X_train, Y_train):
        d, m = X_train.shape
        y = np.array([Y_train])
        w0 = X_train @ y.T / np.sqrt(y @ X_train.T @ X_train @ y.T)
        return w0

    def fit(self, X_train, Y_train):
        alpha, beta, epsilon = self.alpha, self.beta, 1e-8
        max_iter = 200
        w = self.__initialize(X_train, Y_train)
        y = np.array([Y_train])
        d, m = X_train.shape

        if self.boost:
            rho = y @ X_train.T @ w / m
            mask = (1 - (Y_train * (X_train.T @ w).reshape(m)) / rho > 0).reshape(m)
            X = X_train[:, mask]
            A = np.linalg.inv(X @ X.T / (m * beta) + np.identity(d))
            mask_old = mask

        if self.quiet:
            iteration_range = range(max_iter)
        else:
            iteration_range = tqdm(range(max_iter))
        for i in iteration_range:
            # maximize average margin
            # min_w' -yXw/m + alpha||w'-w||^2

            # minimize margin variance
            # min_w' 1/m * sum(max(0, 1 - yixi^Tw / rho)^2) + beta||w'-w||^2
            rho = y @ X_train.T @ w / m
            mask = (1 - (Y_train * (X_train.T @ w).reshape(m)) / rho > 0).reshape(m)
            X = X_train[:, mask]
            s = X @ (Y_train.reshape((m, 1))[mask, :])

            if self.boost:
                xor = np.logical_xor(mask, mask_old)
                new = np.logical_and(mask, xor)
                erase = np.logical_and(mask_old, xor)
                X_new = X_train[:, new]
                X_del = X_train[:, erase]
                for j in range(X_new.shape[1]):
                    t = X_new[:, [j]].T @ A
                    A = A - (t.T @ t) / (m * beta + t @ X_new[:, [j]])
                for j in range(X_del.shape[1]):
                    t = X_del[:, [j]].T @ A
                    A = A - (t.T @ t) / (t @ X_del[:, [j]] - m * beta)
                mask_old = mask
                w_next = A @ (w_next + rho * s / (m * beta))
            else:
                w_next = np.linalg.inv(X @ X.T / (m * beta) + np.identity(d)) @ (w + rho * s / (m * beta))

            rho = y @ X_train.T @ w_next / m
            if rho < 0:
                # print('Iter %d: Average margin below zero, alpha = %.6f, beta = %.6f' % (i, self.alpha, self.beta))
                w_next = -w_next

            w_next = w_next + np.array([np.sum(np.multiply(Y_train, X_train), axis=1)]).T / (2 * alpha * m)
            w_next = w_next / np.linalg.norm(w_next)

            dis = np.linalg.norm(w - w_next) / np.sqrt(d)
            if dis < epsilon:
                break
            else:
                w = w_next
        self.w = w

    def predict(self, x):
        return self.w.T @ x

    def acc(self, X_test, Y_test):
        m = X_test.shape[1]
        result_margin_squared = self.predict(X_test)
        acc = 0
        for i in range(m):
            if result_margin_squared[0][i] * Y_test[i] >= 0:
                acc += 1
        acc = acc / m
        return acc
