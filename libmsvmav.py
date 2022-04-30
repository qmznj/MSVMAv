import numpy as np
from tqdm import tqdm
import numba as nb


@nb.njit(parallel=True)
def calculate_RBF_K(X, gamma):
    m = X.shape[1]
    K = np.zeros((m, m))
    for i in nb.prange(m):
        for k in nb.prange(i):
            K[i][k] = np.exp(-np.sum(np.square(X[:, i] - X[:, k])) * gamma)
            K[k][i] = K[i][k]
        K[i][i] = 1
    return K


@nb.njit(parallel=True)
def calculate_RBF_K_test(X, X_test, gamma):
    num_test = X_test.shape[1]
    m = X.shape[1]
    K = np.zeros((num_test, m))
    for i in nb.prange(num_test):
        for k in nb.prange(m):
            K[i][k] = np.exp(-np.sum(np.square(X_test[:, i] - X[:, k])) * gamma)
    return K


class MSVMAv:
    def __init__(self, alpha, beta, kernel='Linear', gamma=None):
        self.alpha = alpha
        self.beta = beta
        self.kernel = kernel
        self.boost = False
        if self.kernel == 'Linear':
            self.w = None
        elif self.kernel == 'Gaussian':
            self.a = None
            self.gamma = gamma
            self.X = None

    def fit(self, X, Y):
        if self.kernel == 'Linear':
            # Augment for bias term
            X_train = np.vstack((X.T, np.ones((1, X.shape[0]))))
            y = np.array([Y])

            epsilon = 1e-8
            max_iter = 100
            d, m = X_train.shape

            # Initialization
            w = X_train @ y.T / np.sqrt(y @ X_train.T @ X_train @ y.T)

            if self.boost:
                rho = y @ X_train.T @ w / m
                mask = (1 - (Y * (X_train.T @ w).reshape(m)) / rho > 0).reshape(m)
                X = X_train[:, mask]
                A = np.linalg.inv(X @ X.T / (m * self.beta) + np.identity(d))
                mask_old = mask

            for i in range(max_iter):
                # Minimize margin semi-variance
                rho = y @ X_train.T @ w / m
                mask = (1 - (Y * (X_train.T @ w).reshape(m)) / rho > 0).reshape(m)
                X = X_train[:, mask]
                s = X @ (Y.reshape((m, 1))[mask, :])

                if self.boost:
                    xor = np.logical_xor(mask, mask_old)
                    new = np.logical_and(mask, xor)
                    erase = np.logical_and(mask_old, xor)
                    X_new = X_train[:, new]
                    X_del = X_train[:, erase]
                    for j in range(X_new.shape[1]):
                        t = X_new[:, [j]].T @ A
                        A = A - (t.T @ t) / (m * self.beta + t @ X_new[:, [j]])
                    for j in range(X_del.shape[1]):
                        t = X_del[:, [j]].T @ A
                        A = A - (t.T @ t) / (t @ X_del[:, [j]] - m * self.beta)
                    mask_old = mask
                    w_next = A @ (w_next + rho * s / (m * self.beta))
                else:
                    w_next = np.linalg.inv(X @ X.T / (m * self.beta) + np.identity(d)) @ (w + rho * s / (m * self.beta))

                rho = y @ X_train.T @ w_next / m
                if rho < 0:
                    w_next = -w_next

                # Maximize average margin
                w_next = w_next + np.array([np.sum(np.multiply(Y, X_train), axis=1)]).T / (2 * self.alpha * m)
                w_next = w_next / np.linalg.norm(w_next)

                # Stopping condition
                dis = np.linalg.norm(w - w_next) / np.sqrt(d)
                if dis < epsilon:
                    break
                else:
                    w = w_next
            self.w = w

        elif self.kernel == 'Gaussian':
            X_train = X.T
            self.X = X_train
            y = np.array([Y])

            epsilon = 1e-8
            max_iter = 100
            _, m = X_train.shape

            # Gram Matrix
            K = calculate_RBF_K(X_train, self.gamma)
            a = y.T / np.sqrt(y @ K @ y.T)

            rho = y @ K @ a / m
            mask = (1 - (Y * (K @ a).reshape(m)) / rho > 0).reshape(m)
            K_prime = K[:, mask]
            M = np.linalg.inv((K_prime @ K_prime.T) / (m * self.beta) + K + np.identity(m))
            M_list = []
            M_list.append((mask, M))

            for _ in range(max_iter):
                # Minimize margin semi-variance
                rho = y @ K @ a / m
                mask = (1 - (Y * (K @ a).reshape(m)) / rho > 0).reshape(m)
                K_prime = K[:, mask]
                s = K_prime @ (Y.reshape((m, 1))[mask, :])

                min_cnt = m + 1
                min_candidate = None
                for candidate in M_list:
                    mask_old = candidate[0]
                    xor = np.logical_xor(mask, mask_old)
                    new = np.logical_and(mask, xor)
                    new_cnt = np.sum(new)
                    erase = np.logical_and(mask_old, xor)
                    del_cnt = np.sum(erase)
                    tot = new_cnt + del_cnt
                    if tot < min_cnt:
                        min_cnt = tot
                        min_candidate = candidate

                mask_old = min_candidate[0]
                M = min_candidate[1]
                xor = np.logical_xor(mask, mask_old)
                new = np.logical_and(mask, xor)
                new_cnt = np.sum(new)
                erase = np.logical_and(mask_old, xor)
                del_cnt = np.sum(erase)
                K_new = K[:, new]
                K_del = K[:, erase]

                if new_cnt != 0:
                    tmp1 = M @ K_new
                    M = M - tmp1 @ np.linalg.inv(K_new.T @ tmp1 + m * self.beta * np.identity(new_cnt)) @ K_new.T @ M
                if del_cnt != 0:
                    tmp2 = M @ K_del
                    M = M - tmp2 @ np.linalg.inv(K_del.T @ tmp2 - m * self.beta * np.identity(del_cnt)) @ K_del.T @ M

                a_next = M @ ((K + np.identity(m)) @ a + s * rho / (m * self.beta))

                if new_cnt + del_cnt >= 50:
                    M_list.append((mask, M))

                rho = y @ K @ a_next / m
                if rho < 0:
                    a_next = -a_next

                # Maximize average margin
                a_next = a_next + y.T / (2 * self.alpha * m)
                a_next = a_next / np.sqrt(a_next.T @ K @ a_next)

                # Stopping condition
                dis = np.linalg.norm(a - a_next) / np.sqrt(m)
                if dis < epsilon:
                    break
                else:
                    a = a_next
            self.a = a

    def predict(self, x):
        if len(x.shape) == 1:
            m = 1
            X_test = x.reshape((1, x.shape[0]))
        else:
            m = x.shape[0]
            X_test = x.T
        if self.kernel == 'Linear':
            X_test = np.vstack((X_test, np.ones((1, X_test.shape[1]))))
            result = self.w.T @ X_test
        elif self.kernel == 'Gaussian':
            K = calculate_RBF_K_test(X_test, self.X, self.gamma)
            result = self.a.T @ K
        for i in range(m):
            if result[0][i] > 0:
                result[0][i] = 1
            else:
                result[0][i] = -1
        return result.reshape(m)

    def acc(self, X_test, Y_test):
        result = self.predict(X_test)
        return np.sum(result == Y_test) / Y_test.shape[0]
