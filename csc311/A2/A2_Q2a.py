import numpy as np
import matplotlib.pyplot as plt
from run_knn import run_knn
from logistic_regression_template import run_logistic_regression

# (a)
train = np.load("mnist_train.npz")
# print(train.files)
train_data = train['train_inputs']
# print(train_data.shape)
train_labels = train['train_targets']

valid = np.load("mnist_valid.npz")
# print(valid.files)
valid_data = valid['valid_inputs']
# print(valid_data.shape)
valid_labels = valid['valid_targets']

test = np.load("mnist_test.npz")
# print(test.files)
test_data = test['test_inputs']
# print(test_data.shape)
test_labels = test['test_targets']


classification_rate_validation = []
classification_rate_test = []
for k in [1, 3, 5, 7, 9]:
    vk_labels = run_knn(k, train_data, train_labels, valid_data)
    tk_labels = run_knn(k, train_data, train_labels, test_data)
    classification_rate_validation.append(np.count_nonzero(vk_labels == valid_labels) / len(valid_labels))
    classification_rate_test.append(np.count_nonzero(tk_labels == test_labels) / len(test_labels))

print(classification_rate_validation)
print(classification_rate_test)

plt.scatter(np.array([1, 3, 5, 7, 9]), classification_rate_validation)
plt.scatter(np.array([1, 3, 5, 7, 9]), classification_rate_test)
plt.legend(['Classification rate of validation data', 'Classification rate of test data'])
plt.show()



