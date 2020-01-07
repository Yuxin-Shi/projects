import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt


num_samples = 400
cov = np.array([[1., .7], [.7, 1.]]) * 10
mean_1 = [.1, .1]
mean_2 = [6., .1]

x_class1 = np.random.multivariate_normal(mean_1, cov, num_samples // 2)
x_class2 = np.random.multivariate_normal(mean_2, cov, num_samples // 2)
xy_class1 = np.column_stack((x_class1, np.zeros(num_samples // 2)))
xy_class2 = np.column_stack((x_class2, np.ones(num_samples // 2)))
data_full = np.row_stack([xy_class1, xy_class2])
np.random.shuffle(data_full)
data = data_full[:, :2]
labels = data_full[:, 2]

plt.scatter(x_class1[:, 0], x_class1[:, 1], marker="x")  # first class, x shape
plt.scatter(x_class2[:, 0], x_class2[:, 1], marker="o")  # second class, circle shape
plt.show()


def cost(data, R, Mu):
    N, D = data.shape
    K = Mu.shape[1]
    J = 0
    for k in range(K):
        J += np.dot(np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2, R[:, k])
    return J


def normal_density(x, mu, Sigma):
    return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
        / np.sqrt(np.linalg.det(2 * np.pi * Sigma))


def log_likelihood(data, Mu, Sigma, Pi):
    """ Compute log likelihood on the data given the Gaussian Mixture Parameters.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        L: a scalar denoting the log likelihood of the data given the Gaussian Mixture
    """
    # Fill this in:
    N, D = data.shape[0], data.shape[1]  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of mixtures
    L, T = 0., 0.
    for n in range(N):
        T = 0
        for k in range(K):
            T += Pi[k] * normal_density(data[n], Mu[:, k], Sigma[k])
            # Compute the likelihood from the k-th Gaussian weighted by the mixing coefficients
        L += np.log(T)
    return L


# TODO: Gaussian Mixture Expectation Step
def gm_e_step(data, Mu, Sigma, Pi):
    """ Gaussian Mixture Expectation Step.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        Gamma: a NxK matrix of responsibilities
    """
    # Fill this in:
    N, D = data.shape[0], data.shape[1]  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of mixtures
    Gamma = np.zeros((N,K))  # zeros of shape (N,K), matrix of responsibilities
    for n in range(N):
        for k in range(K):
            Gamma[n, k] = Pi[k] * normal_density(data[n], Mu[:, k], Sigma[k])
        Gamma[n, :] /= np.sum(Gamma, axis=1)[n]
        # Normalize by sum across second dimension (mixtures)
    return Gamma


# TODO: Gaussian Mixture Maximization Step
def gm_m_step(data, Gamma):
    """ Gaussian Mixture Maximization Step.

    Args:
        data: a NxD matrix for the data points
        Gamma: a NxK matrix of responsibilities

    Returns:
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients
    """
    # Fill this in:
    N, D = data.shape[0], data.shape[1]  # Number of datapoints and dimension of datapoint
    K = Gamma.shape[1]  # number of mixtures
    Nk = sum(Gamma)  # Sum along first axis
    Mu = data.T.dot(Gamma) / Nk
    Sigma = [0] * K
    for k in range(K):
        gamma_matrix = np.diag(Gamma[:, k])
        Sigma[k] = (data - Mu[:, k]).T.dot(gamma_matrix).dot(data - Mu[:, k]) / Nk[k]
    Pi = Nk / N
    return Mu, Sigma, Pi


# TODO: Run this cell to call the Gaussian Mixture EM algorithm
N, D = data.shape
K = 2
Mu = np.zeros([D, K])
Mu[:, 1] = 1.
Sigma = [np.eye(2), np.eye(2)]
Pi = np.ones(K) / K
Gamma = np.zeros([N, K])  # Gamma is the matrix of responsibilities

max_iter = 200

costs = []
log_likelihoods = []
for it in range(max_iter):
    Gamma = gm_e_step(data, Mu, Sigma, Pi)
    Mu, Sigma, Pi = gm_m_step(data, Gamma)
    costs.append(cost(data, Gamma, Mu))
    log_likelihoods.append(log_likelihood(data, Mu, Sigma, Pi))
#     print(it, log_likelihood(data, Mu, Sigma, Pi))  # This function makes the computation longer, but good for debugging
# print(Gamma)

class_1 = np.where(Gamma[:, 0] >= .5)
class_2 = np.where(Gamma[:, 1] >= .5)

class_1_labels = labels[class_1[0]]
match_1 = max(sum(class_1_labels), len(class_1[0]) - sum(class_1_labels))
class_2_labels = labels[class_2[0]]
match_2 = max(sum(class_2_labels), len(class_2[0]) - sum(class_2_labels))
accuracy = (match_1 + match_2) / N
print("Misclassification error is ", 1 - accuracy)

# TODO: Make a scatterplot for the data points showing the Gaussian Mixture cluster assignments of each point
plt.scatter(data[class_1[0], np.zeros(class_1[0].shape[0], dtype=int)],
            data[class_1[0], np.ones(class_1[0].shape[0], dtype=int)], marker="x")  # first class, x shape
plt.scatter(data[class_2[0], np.zeros(class_2[0].shape[0], dtype=int)],
            data[class_2[0], np.ones(class_2[0].shape[0], dtype=int)], marker="o")  # second class, circle shape
plt.show()

plt.plot(np.arange(max_iter), log_likelihoods)
plt.show()
