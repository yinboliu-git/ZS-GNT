import csv
import os

import joblib
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

# 读取数据
# data = pd.read_excel('result_with_drugs.xlsx')  # 更改为实际的文件路径

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

import numpy as np
import numpy as np
from multiprocessing import Pool

def calculate_metrics_for_threshold(args):
    predict_score, real_score, threshold = args
    prediction = (predict_score >= threshold).astype(int)
    TP = np.sum((prediction == 1) & (real_score == 1))
    FP = np.sum((prediction == 1) & (real_score == 0))
    FN = np.sum((prediction == 0) & (real_score == 1))
    TN = np.sum((prediction == 0) & (real_score == 0))
    return TP, FP, FN, TN,threshold

def get_metrics(real_score, predict_score,num_processes=16):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.linspace(0.001, 0.999, 998))]

    with Pool(num_processes) as pool:
        metrics = pool.map(calculate_metrics_for_threshold, [(predict_score, real_score, th) for th in thresholds])

    TP, FP, FN, TN, threshold = zip(*metrics)
    TP, FP, FN, TN = np.array(TP), np.array(FP), np.array(FN), np.array(TN),
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    # print(
    #     ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(
    #         auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [ real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


if __name__ == '__main__':
    l = np.random.randint(0, 2, size=10000)
    p = np.random.randint(0, 2, size=10000)  # Use a realistic random prediction
    results, names = get_metrics(l, p)
    print(dict(zip(names, results)))
