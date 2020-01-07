import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt

# TODO: Run this cell to generate the data
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

# TODO: Make a scatterplot for the data points showing the true cluster assignments of each point
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


# TODO: K-Means Assignment Step
def km_assignment_step(data, Mu):
    """ Compute K-Means assignment step

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the cluster means locations

    Returns:
        R_new: a NxK matrix of responsibilities
    """

    # Fill this in:
    N, D = data.shape  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of clusters
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = np.linalg.norm(data - Mu[:, k], axis=1)
    arg_min = np.argmin(r, axis=1)  # argmax/argmin along dimension 1
    R_new = np.zeros((N, K))  # Set to zeros/ones with shape (N, K)
    R_new[np.arange(N), arg_min] = 1  # Assign to 1
    return R_new


# TODO: K-means Refitting Step
def km_refitting_step(data, R, Mu):
    """ Compute K-Means refitting step.

    Args:
        data: a NxD matrix for the data points
        R: a NxK matrix of responsibilities
        Mu: a DxK matrix for the cluster means locations

    Returns:
        Mu_new: a DxK matrix for the new cluster means locations
    """
    N, D = data.shape # Number of datapoints and dimension of datapoint
    K = R.shape[1]  # number of clusters
    Mu_new = data.T.dot(R) / sum(R)
    return Mu_new


# TODO: Run this cell to call the K-means algorithm
N, D = data.shape
K = 2
max_iter = 100
class_init = np.random.binomial(1., .5, size=N)
R = np.vstack([class_init, 1 - class_init]).T

Mu = np.zeros([D, K])
Mu[:, 1] = 1.
R.T.dot(data), np.sum(R, axis=0)

costs = []
for it in range(max_iter):
    R = km_assignment_step(data, Mu)
    Mu = km_refitting_step(data, R, Mu)
    costs.append(cost(data, R, Mu))


class_1 = np.where(R[:, 0])
class_2 = np.where(R[:, 1])

class_1_labels = labels[class_1[0]]
match_1 = max(sum(class_1_labels), len(class_1[0]) - sum(class_1_labels))
class_2_labels = labels[class_2[0]]
match_2 = max(sum(class_2_labels), len(class_2[0]) - sum(class_2_labels))
accuracy = (match_1 + match_2) / N
print("Misclassification error is ", 1 - accuracy)

# TODO: Make a scatterplot for the data points showing the K-Means cluster assignments of each point
plt.scatter(data[class_1[0], np.zeros(class_1[0].shape[0], dtype=int)],
            data[class_1[0], np.ones(class_1[0].shape[0], dtype=int)], marker="x")  # first class, x shape
plt.scatter(data[class_2[0], np.zeros(class_2[0].shape[0], dtype=int)],
            data[class_2[0], np.ones(class_2[0].shape[0], dtype=int)], marker="o")  # second class, circle shape
plt.show()
plt.plot(np.arange(max_iter), costs)
plt.show()

