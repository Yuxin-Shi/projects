from logistic_regression_template import *
import matplotlib.pyplot as plt

list_train_error_avg = []
list_train_ce_avg = []
list_valid_error_avg = []
list_valid_ce_avg = []
list_lmbda = [0, 0.001, 0.01, 0.1, 1]
for lmbda in list_lmbda:
    train_error_avg, train_ce_avg,\
        valid_error_avg, valid_ce_avg = run_pen_logistic_regression(lmbda)
    list_train_error_avg.append(train_error_avg)
    list_train_ce_avg.append(train_ce_avg)
    list_valid_error_avg.append(valid_error_avg)
    list_valid_ce_avg.append(valid_ce_avg)

fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(np.array(list_lmbda), list_train_error_avg)
ax1.scatter(np.array(list_lmbda), list_valid_error_avg)
ax1.legend(['Classification error of train data', 'Classification error of validation data'])
ax2.scatter(np.array(list_lmbda), list_train_ce_avg)
ax2.scatter(np.array(list_lmbda), list_valid_ce_avg)
ax2.legend(['Cross entropy of train data', 'Cross entropy of validation data'])
plt.show()

run_pen_logistic_regression(0.001)