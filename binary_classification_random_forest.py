import torch
import numpy as np
import os, warnings
from sklearn.metrics import r2_score
from torch.autograd import Variable
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from utils import *
import scipy.stats as stats
import IPython

def prepare_regression_data(data, number_of_history_vfs):
    
    # Split based on the patients
    pid_indices = np.unique(data.pid)
    pid_train, pid_test = train_test_split(pid_indices, test_size=0.15, random_state=42)
    pid_train, pid_val = train_test_split(pid_train, test_size=0.15, random_state=30)
    
    indices_train = np.where(np.isin(data.pid, pid_train))[0]
    indices_val = np.where(np.isin(data.pid, pid_val))[0]
    indices_test = np.where(np.isin(data.pid, pid_test))[0]

    if number_of_history_vfs > 1:
        N, _, num_locs = data.td.shape
        data.td = data.td.reshape(N, number_of_history_vfs*num_locs)

    # td_diff = np.diff(data.vf, axis=1).squeeze()
    # Input
    # x_train = td_diff[indices_train]
    # x_val = td_diff[indices_val]
    # x_test = td_diff[indices_test]
    x_train = data.td[indices_train]
    x_val = data.td[indices_val]
    x_test = data.td[indices_test]

    # targets
    y_train = data.labels_binary[indices_train]
    y_val = data.labels_binary[indices_val]
    y_test = data.labels_binary[indices_test]

    # Add current age and time difference to the input
    if number_of_history_vfs == 1:
        data.age_curr = data.age_curr[:, None]
        data.md_curr = data.md_curr[:, None]
    time_diffs = data.age_next - data.age_curr[:, -1]
    x_train = np.concatenate([x_train, data.age_curr[indices_train], time_diffs[indices_train, None]], axis=1)
    x_val = np.concatenate([x_val, data.age_curr[indices_val], time_diffs[indices_val, None]], axis=1)
    x_test = np.concatenate([x_test, data.age_curr[indices_test], time_diffs[indices_test, None]], axis=1)

    x_train = np.concatenate([x_train, data.md_curr[indices_train]], axis=1)
    x_val = np.concatenate([x_val, data.md_curr[indices_val]], axis=1)
    x_test = np.concatenate([x_test, data.md_curr[indices_test]], axis=1)

    # scaler = StandardScaler() # MinMaxScaler(feature_range=[-1, 1])
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, indices_train, indices_val, indices_test

def run_extra_trees(input_tr, output_tr, input_test, output_test):

    clf_et = RandomForestClassifier(n_estimators=10000, min_impurity_decrease=0, class_weight="balanced")
    clf_et.fit(input_tr, output_tr)
    pred_clf_et = clf_et.predict_proba(input_test)[:, 1]

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(output_test, pred_clf_et, pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    prec, recall, thresholds_pr = precision_recall_curve(output_test, pred_clf_et, pos_label=1)
    area_under_prec = auc(recall, prec)
    print("AUC-ROC: {:.3f}".format(area_under_roc))
    print("AUC-PREC: {:.3f}".format(area_under_prec))

    return area_under_roc, area_under_prec

def run_random_forest(input_tr, output_tr, input_test, output_test):
    
    clf_rf = RandomForestClassifier(n_estimators=10000, min_impurity_decrease=0, class_weight="balanced")
    clf_rf.fit(input_tr, output_tr)
    pred_clf_rf = clf_rf.predict_proba(input_test)[:, 1]

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(output_test, pred_clf_rf, pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    prec, recall, thresholds_pr = precision_recall_curve(output_test, pred_clf_rf, pos_label=1)
    area_under_prec = auc(recall, prec)
    print("AUC-ROC: {:.3f}".format(area_under_roc))
    print("AUC-PREC: {:.3f}".format(area_under_prec))

    return area_under_roc, area_under_prec

def main():
    
    # DEFS
    dataset_name = 'Rotterdam'
    # number_of_required_vfs = 5
    number_of_required_vfs_range = [3]

    aucs_roc = []
    aucs_prec = []
    for number_of_required_vfs in number_of_required_vfs_range:
        # Get the data
        data = get_longitudinal_data(dataset_name, number_of_required_vfs=number_of_required_vfs, use_all_samples=False)

        # Split the data
        input_tr, output_tr, input_val, output_val, input_test, \
            output_test, ind_tr, ind_val, ind_test = prepare_regression_data(data, number_of_history_vfs=number_of_required_vfs)


        # Get only MD and time inputs
        # input_tr = input_tr[:, -1*(2*number_of_required_vfs+1):]
        # input_val = input_val[:, -1*(2*number_of_required_vfs+1):]
        # input_test = input_test[:, -1*(2*number_of_required_vfs+1):]

        # Combine traning and validation sets
        new_input = np.concatenate([input_tr, input_val], axis=0)
        new_output = np.concatenate([output_tr, output_val], axis=0)

        # Print some info
        print("Data set: {}".format(dataset_name))
        print("Number of previous VFs: {:d}".format(number_of_required_vfs))
        print("Number of training samples: {:d}".format(input_tr.shape[0]))
        print("Number of validation samples: {:d}".format(input_val.shape[0]))
        print("Number of test samples: {:d}".format(input_test.shape[0]))
        print("Number of pos./neg. samples in training:{:d}/{:d}".format(new_output.sum(), (new_output==0).sum()))
        print("Number of pos./neg. samples in test:{:d}/{:d}".format(output_test.sum(), (output_test==0).sum()))

        # Classification
        auc_roc, auc_prec = run_random_forest(new_input, new_output, input_test, output_test)
        aucs_roc.append(auc_roc)
        aucs_prec.append(auc_prec)

    plt.plot(number_of_required_vfs_range, aucs_roc, label='AUC-ROC')
    plt.plot(number_of_required_vfs_range, aucs_prec, label='AP')


    IPython.embed()

if __name__ == "__main__":
    main()

