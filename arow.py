# Copyright 2018 Nadheesh Jihan
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
References 
[1] AROW - http://www.alexkulesza.com/pubs/arow_nips09.pdf
[2] MOA - http://www.jmlr.org/papers/volume11/bifet10a/bifet10a.pdf
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

seed = 42
np.random.seed(seed)


class AROW:

    def __init__(self, nb_class, d):
        self.w = np.zeros((nb_class, d))
        self.sigma = np.identity(d)
        self.r = 1
        self.nb_class = nb_class

    def fit(self, X, y):
        w = np.copy(self.w)
        sigma = np.copy(self.sigma)

        # y = ((y - y.min()) * (1/(y.max() - y.min()) * (nb_class-1))).astype('uint8')

        F_t = np.dot(self.w, X.T)

        # compute hinge loss and support vector
        F_s = np.copy(F_t)
        F_s[y] = -np.inf
        s_t = np.argmax(F_s)
        m_t = F_t[y] - F_t[s_t]
        v_t = np.dot(X, np.dot(sigma, X.T))
        l_t = np.maximum(0, 1 - m_t)  # hinge loss

        # update weights
        if l_t > 0:
            beta_t = 1 / (v_t + self.r)
            alpha_t = l_t * beta_t
            self.w[y] = w[y] + (alpha_t * np.dot(sigma, X.T).T)
            self.w[s_t] = w[s_t] - (alpha_t * np.dot(sigma, X.T).T)
            self.sigma = sigma - beta_t * np.dot(np.dot(sigma, X.T), np.dot(X, sigma))

    def predict(self, X):
        return np.argmax(np.dot(self.w, X.T), axis=0)


from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris = load_iris()

    X_train_iris, y_train_iris = shuffle(iris.data, iris.target, random_state=seed)

    # X_train_iris = MinMaxScaler().fit_transform(X_train_iris)

    n, d = X_train_iris.shape
    nb_class = len(set(iris.target))

    arow = AROW(nb_class, d)

    error = 0
    for i in range(n):
        X, y = X_train_iris[i:i + 1], y_train_iris[i:i + 1]

        p_y = arow.predict(X)
        arow.fit(X, y)

        if y-p_y != 0:
            error += 1

    print(error)
    print(np.divide(error, n, dtype=np.float))
