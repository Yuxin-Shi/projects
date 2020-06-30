import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


def black_litterman(tau, P, Q, Pi, Sigma):
    Omega = np.diag(np.diag(P.dot(tau * Sigma).dot(P.T)))
    H = np.linalg.inv(tau * Sigma) + P.T.dot(np.linalg.inv(Omega)).dot(P)
    C = np.linalg.inv(tau * Sigma).dot(Pi) + P.T.dot(np.linalg.inv(Omega)).dot(Q)

    bl_mean = np.linalg.inv(H).dot(C)
    bl_cov = Sigma + np.linalg.inv(H)

    return bl_mean, bl_cov


def optimized_weight(mean, cov, delta):
    n = mean.shape[0]
    x0 = np.ones(n) / n
    bounds = tuple(tuple((0, 1) for x in range(n)))
    constraints = {'type': 'eq',
                   'fun': lambda w: sum(w) - 1,
                   'jac': lambda w: np.ones(mean.shape)}
    pram = (mean, cov, delta)

    def func(w, mean, cov, delta):
        return -w.T.dot(mean) + delta / 2 * w.T.dot(cov).dot(w)

    def func_der(w, mean, cov, delta):
        return -mean + delta * cov.dot(w)

    res = optimize.minimize(fun=func, args=pram, x0=x0, method='SLSQP',
                            jac=func_der, constraints=constraints
                            , options={'ftol': 1e-12, 'disp': True}, bounds=bounds)

    return res.x


if __name__ == '__main__':
    df1 = pd.read_excel("C:/University/2020 Summer/BL回测/back_test.xlsx")
    df2 = pd.read_excel("C:/University/2020 Summer/BL回测/w_eq_helper.xlsx")

    history_data_start_line = np.array([31, 35, 43, 47, 54, 59, 78, 83, 92,
                                       101, 114, 125, 128, 137, 145, 167, 177])
    history_data_end_line = history_data_start_line + 24
    data_period_start_line = history_data_end_line
    data_period_end_line = np.append(data_period_start_line[1:], df1.values.shape[0])
    # [71, 79, 83, 90, 95, 114, 119, 128, 137, 150, 161, 164, 173, 181, 203, 213, 215]

    P_dict = {1: np.array([[1, 0, 0], [0, -1, 1]]),
              2: np.array([[0, 0, 1], [-1, 1, 0]]),
              3: np.array([[0, 0, 1], [1, -1, 0]]),
              4: np.array([[0, 1, 0], [1, 0, -1]])}
    Q_dict = {1: np.array([0.61, 0.017]),
              2: np.array([0.074, 0.057]),
              3: np.array([0.099, 0.014]),
              4: np.array([0.088, 0.072])}

    bl_returns = np.array([])
    mv_returns = np.array([])

    num_period = history_data_start_line.shape[0]
    # 改下面这两行可以控制图像的起始周期和结束周期
    start_period_num = 1
    end_period_num = 17
    for i in range(start_period_num-1, end_period_num):
        history_data = df1.values[history_data_start_line[i]:history_data_end_line[i],
                                  10:13]

        ratio = df2.values[i][1]
        denominator = ratio + 1 + 1/3
        w_eq = np.array([ratio / denominator, 1 / denominator, 1 / (3*denominator)])

        tau = 1 / 24
        P = P_dict[df1.values[data_period_start_line[i]][6]]
        Q = Q_dict[df1.values[data_period_start_line[i]][6]]
        Sigma = np.cov(history_data.T.astype(float))
        delta = 2.5
        Pi = delta * Sigma.dot(w_eq)

        bl_mean, bl_cov = black_litterman(tau, P, Q, Pi, Sigma)
        weights_bl = optimized_weight(bl_mean, bl_cov, delta)
        weights = optimized_weight(np.mean(history_data, axis=0), Sigma, delta)

        if i == start_period_num-1:
            weights_bl_matrix = weights_bl
            weights_matrix = weights
        else:
            weights_bl_matrix = np.vstack([weights_bl_matrix, weights_bl])
            weights_matrix = np.vstack([weights_matrix, weights])

        data_period = df1.values[data_period_start_line[i]:data_period_end_line[i],
                                 10:13]
        bl_return = data_period.dot(weights_bl)
        mv_return = data_period.dot(weights)
        bl_returns = np.append(bl_returns, bl_return)
        mv_returns = np.append(mv_returns, mv_return)

        stock_returns = df1.values[data_period_start_line[start_period_num-1]:
                                   data_period_end_line[end_period_num-1], 10]
        security_returns = df1.values[data_period_start_line[start_period_num-1]:
                                      data_period_end_line[end_period_num-1], 11]
        future_returns = df1.values[data_period_start_line[start_period_num-1]:
                                    data_period_end_line[end_period_num-1], 12]

    register_matplotlib_converters()

    plt.plot(df1.values[data_period_start_line[start_period_num-1]:
                        data_period_end_line[end_period_num-1], 0],
             bl_returns)
    plt.plot(df1.values[data_period_start_line[start_period_num-1]:
                        data_period_end_line[end_period_num-1], 0],
             mv_returns)
    plt.legend(['BL Returns', 'Mean_Variance Returns'])
    plt.show()

    plt.plot(df1.values[data_period_start_line[start_period_num - 1]:
                        data_period_end_line[end_period_num - 1], 0],
             bl_returns)
    plt.plot(df1.values[data_period_start_line[start_period_num-1]:
                        data_period_end_line[end_period_num-1], 0],
             stock_returns)
    plt.plot(df1.values[data_period_start_line[start_period_num - 1]:
                        data_period_end_line[end_period_num - 1], 0],
             security_returns)
    plt.plot(df1.values[data_period_start_line[start_period_num - 1]:
                        data_period_end_line[end_period_num - 1], 0],
             future_returns)
    plt.legend(['BL Returns', 'Stock Returns', 'Security Returns', 'Future Returns'])
    plt.show()

    print(weights_bl_matrix)
    print(weights_matrix)









