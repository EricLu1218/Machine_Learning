'''
NTHU EE Machine Learning HW2
Author: 呂宸漢
Student ID: 110062802
'''
import argparse
import pandas as pd
import numpy as np


def get_gaussian_basis(X: np.ndarray, mu: float, sigma: float) -> float:
        return np.exp(-((X - mu) ** 2) / (2 * sigma ** 2))


def get_feature_vector(features: np.ndarray, O_1: int, O_2: int) -> np.ndarray:
    X_1 = features[:, 0]
    X_2 = features[:, 1]
    X_3 = features[:, 2]

    s_1 = (np.max(X_1) - np.min(X_1)) / (O_1 - 1)
    s_2 = (np.max(X_2) - np.min(X_2)) / (O_2 - 1)

    phi = np.ones((features.shape[0], O_1 * O_2 + 2))
    for i in range(1, O_1 + 1):
        for j in range(1, O_2 + 1):
            mu_i = s_1 * (i - 1) + np.min(X_1)
            mu_j = s_2 * (j - 1) + np.min(X_2)
            k = O_2 * (i - 1) + j
            phi[:, k - 1] = get_gaussian_basis(X_1, mu_i, s_1) * get_gaussian_basis(X_2, mu_j, s_2)
    phi[:, -2] = X_3
    return phi


# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=4):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    train_data_feature = train_data[:, :3]
    train_data_label = train_data[:, 3]

    train_phi = get_feature_vector(train_data_feature, O1, O2)
    weights = np.linalg.inv(np.identity(train_phi.shape[1]) + train_phi.T @ train_phi) @ train_phi.T @ train_data_label
    y_BLR_prediction = get_feature_vector(test_data_feature, O1, O2) @ weights
    return y_BLR_prediction


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=4):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    train_data_feature = train_data[:, :3]
    train_data_label = train_data[:, 3]

    train_phi = get_feature_vector(train_data_feature, O1, O2)
    weights = np.linalg.inv(train_phi.T @ train_phi) @ train_phi.T @ train_data_label
    y_MLLS_prediction = get_feature_vector(test_data_feature, O1, O2) @ weights
    return y_MLLS_prediction


def CalMSE(data, prediction):
    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean_squared_error = sum_squared_error / prediction.shape[0]
    return mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=2)
    parser.add_argument('-O2', '--O_2', type=int, default=4)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()
