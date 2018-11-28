import numpy as np 
import matplotlib.pyplot as plt 
from cvxopt import matrix, solvers

COLOR = ['red', 'blue']
N = 30

def plot_data(x,y):
    uniq = np.unique(y)
    for i in range(len(np.unique(y))):
        x_sub = x[y == uniq[i]]
        plt.scatter(x_sub[:, 0], x_sub[:, 1], c = COLOR[i])

def plot_separator(w, b):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 6)
    plt.plot(x, x * slope + intercept, 'k-')

def fit(V, y):
    K = matrix(V.T.dot(V))
    p = matrix(-np.ones((2*N, 1)))
    G = matrix(-np.eye(2*N))
    h = matrix(np.zeros((2*N, 1)))
    A = matrix(y.reshape(1,-1))
    b = matrix(np.zeros((1, 1))) 
    solvers.options['show_progress'] = False
    sol = solvers.qp(K, p, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def main():
    mean = [[3,4], [2,2]]
    cov = [np.diag(0.3*np.ones((2,))), np.diag(0.2*np.ones((2,)))]
    x1 = np.random.multivariate_normal(mean[0], cov[0], N)
    x2 = np.random.multivariate_normal(mean[1], cov[1], N)

    y1 = np.ones((x1.shape[0]))
    y2 = -np.ones((x2.shape[0]))

    x = np.concatenate((x1, x2), axis = 0)
    y = np.concatenate((y1, y2), axis = 0)

    V = np.concatenate((x1.T, -x2.T), axis = 1)
    alphas = fit(V, y)
    w = np.sum(alphas * y[:, None]* x, axis= 0)
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]
    plot_data(x, y)
    plot_separator(w, bias)
    plt.show()

if __name__ == "__main__":
    main()