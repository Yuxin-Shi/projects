import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
                    'learning_rate': 0.1,
                    'weight_regularization': 0,
                    'num_iterations': 200
                 }

    # Logistic regression weights

    weights = np.random.rand(M+1, 1) / 10

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    TRAIN_CE = []
    VALID_CE = []

    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        f_test, df_test, predictions_test = logistic(weights, test_inputs, test_targets, hyperparameters)
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)
        TRAIN_CE.append(cross_entropy_train[0][0])

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        VALID_CE.append(cross_entropy_valid[0][0])
        
        # print some stats
        print ("ITERATION:{}  TRAIN NLOGL:{}  TRAIN CE:{} "
               "TRAIN FRAC:{}  VALID CE:{}  VALID FRAC:{}".format(
                   t+1, f / N, cross_entropy_train[0][0], frac_correct_train*100,
                   cross_entropy_valid[0][0], frac_correct_valid*100))
        print("ITERATION:{}  TEST NLOGL:{}  TEST CE:{} "
               "TEST FRAC:{}".format(
                   t+1, f_test / test_targets.shape[0],
            cross_entropy_test[0][0], frac_correct_test*100))

    plt.scatter(np.arange(1, hyperparameters['num_iterations'] + 1), TRAIN_CE)
    plt.scatter(np.arange(1, hyperparameters['num_iterations'] + 1), VALID_CE)
    plt.legend(['Cross entropy for train data', 'Cross entropy for validation data'])
    plt.show()


def run_pen_logistic_regression(lmbda):
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        'learning_rate': 0.1,
        'weight_regularization': lmbda,
        'num_iterations': 200
    }

    # Logistic regression weights

    weights = np.random.rand(M + 1, 1) / 10

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    TRAIN_CE = []
    VALID_CE = []
    TRAIN_ERROR = []
    VALID_ERROR = []

    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        f_valid, df_valid, predictions_valid = logistic_pen(weights, valid_inputs, valid_targets, hyperparameters)

        f_test, df_test, predictions_test = logistic_pen(weights, test_inputs,
        test_targets, hyperparameters)
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)
        TRAIN_CE.append(cross_entropy_train[0][0])
        TRAIN_ERROR.append(1 - frac_correct_train)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        VALID_CE.append(cross_entropy_valid[0][0])
        VALID_ERROR.append(1 - frac_correct_valid)

        # print some stats
        # print("ITERATION:{}  TRAIN NLOGL:{}  TRAIN CE:{} "
        #       "TRAIN FRAC:{}  VALID NLOGL:{}  VALID CE:{}  VALID FRAC:{}".format(
        #     t + 1, f / N, cross_entropy_train, frac_correct_train * 100, f_valid / valid_inputs.shape[0],
        #     cross_entropy_valid, frac_correct_valid * 100))
        print("ITERATION:{}  TEST NLOGL:{}  TEST CE:{} "
              "TEST FRAC:{}".format(
            t + 1, f_test[0][0] / test_targets.shape[0],
            cross_entropy_test[0][0], frac_correct_test * 100))

    plt.scatter(np.arange(1, hyperparameters['num_iterations'] + 1), TRAIN_CE)
    plt.scatter(np.arange(1, hyperparameters['num_iterations'] + 1), VALID_CE)
    plt.legend(['Cross entropy for train data with lambda = ' + str(lmbda),
                'Cross entropy for validation data with lambda = ' + str(lmbda)])
    plt.show()

    return np.average(np.array(TRAIN_ERROR)), np.average(np.array(TRAIN_CE)),\
           np.average(np.array(VALID_ERROR)), np.average(np.array(VALID_CE))


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == '__main__':
    run_logistic_regression()
