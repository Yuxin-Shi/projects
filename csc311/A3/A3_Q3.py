from utils import *
import matplotlib.pyplot as plt


def project_to_train(k, inputs_data, inputs_train_data):

    mean_train = np.mean(inputs_train_data, axis=0)
    cov_train = (inputs_train_data - mean_train).T.dot((inputs_train_data - mean_train))

    e_values, e_vectors = np.linalg.eig(cov_train)
    max_k_e_vectors = e_vectors.T[np.argsort(e_values)[-k:]]

    centered_inputs_train = inputs_train_data - mean_train
    centered_inputs = inputs_data - mean_train

    projection_input = centered_inputs.dot(max_k_e_vectors.T)
    projection_train = centered_inputs_train.dot(max_k_e_vectors.T)

    return projection_input, projection_train


def l2_distance(a, b):

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)


def run_1nn(train_data, train_labels, valid_data):
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :1]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    # valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def accuracy(predict_labels, actual_labels):
    acc = np.mean(predict_labels == actual_labels)

    return acc


if __name__ == '__main__':
    inputs_train, inputs_valid, inputs_test, target_train, \
        target_valid, target_test = load_data("digits.npz")

    valid_accuracy = []
    k_values = [2, 5, 10, 20, 30]
    for i in k_values:
        project_valid, project_train = project_to_train(i, inputs_valid, inputs_train)
        predict_valid_labels = run_1nn(project_train, target_train, project_valid)
        validation_accuracy = accuracy(predict_valid_labels, target_valid)
        valid_accuracy.append(validation_accuracy)

        print("Validation set classification accuracy for k =", i, " is ", validation_accuracy)

    plt.plot(np.array([2, 5, 10, 20, 30]), 1 - np.array(valid_accuracy))
    plt.legend(['Classification error of validation data'])
    plt.show()

    for j in k_values:
        project_test, project_train = project_to_train(j, inputs_test, inputs_train)
        predict_test_labels = run_1nn(project_train, target_train, project_test)
        test_accuracy = accuracy(predict_test_labels, target_test)

        print("Test set classification accuracy for k =", j, " is ", test_accuracy)
