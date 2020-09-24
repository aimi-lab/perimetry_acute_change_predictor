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

    # MD targets
    output_scale_factor = 1
    md_diffs = data.md_next - data.md_curr[:, -1]
    y_train = md_diffs[indices_train] * output_scale_factor
    y_val = md_diffs[indices_val] * output_scale_factor
    y_test = md_diffs[indices_test] * output_scale_factor

    # Add current age and time difference to the input
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


def main():
    
    # DEFS
    dataset_name = 'Bern'
    data_filename = "./data_{}.pkl".format(dataset_name.lower())
    number_of_required_vfs = 5

    data = get_longitudinal_data(dataset_name, number_of_required_vfs=number_of_required_vfs) #load_object(data_filename)
    d_in = data.td.shape[1] + 3 # Age + time diffs + md

    # Split the data
    input_tr, output_tr, input_val, output_val, input_test, \
        output_test, ind_tr, ind_val, ind_test = prepare_regression_data(data, number_of_history_vfs=number_of_required_vfs)

    input_tr = input_tr[:, -1*(2*number_of_required_vfs+1):]
    input_val = input_val[:, -1*(2*number_of_required_vfs+1):]
    input_test = input_test[:, -1*(2*number_of_required_vfs+1):]

    # Print some info
    print("Data set: {}".format(dataset_name))
    print("Number of previous VFs: {:d}".format(number_of_required_vfs))
    print("Number of training samples: {:d}".format(input_tr.shape[0]))
    print("Number of validation samples: {:d}".format(input_val.shape[0]))
    print("Number of test samples: {:d}".format(input_test.shape[0]))

    # # Regression forest
    new_input = np.concatenate([input_tr, input_val], axis=0)
    new_output = np.concatenate([output_tr, output_val], axis=0)

    r2s = []
    reg_rf = RandomForestRegressor(n_estimators=10000, min_impurity_decrease=0)
    # parameters = {"n_estimators":[10, 100, 500], "max_depth":np.arange(10, 100)}
    # clf_rf_opt = GridSearchCV(clf_rf, parameters, cv=5)
    # clf_rf_opt.fit(new_input, new_output)
    reg_rf.fit(new_input, new_output)
    pred_reg_rf = reg_rf.predict(input_test)
    
    # r2 score
    r2score = r2_score(output_test, pred_reg_rf)
    r, p = stats.spearmanr(output_test, pred_reg_rf)
    print(r2score)
    print(r)
    r2s.append(r2score)

    plt.plot(min_impurity_dec_range, r2s)
    mse = np.mean((output_test-pred_rf) **2)

    max_val = max(np.max(output_test), np.max(pred_rf))
    min_val = min(np.min(output_test), np.min(pred_rf))
    plt.plot(output_test, pred_rf, '*', label="r2={:.3f}".format(r2score))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.legend()
    plt.xlabel("True MD change")
    plt.ylabel("Predicted MD change")
    plt.grid()
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.savefig('results/md_diff_regression/true_vs_pred_md_diff_{}.pdf'.format(dataset_name.lower()))
    plt.show()
        
    IPython.embed()

if __name__ == "__main__":
    main()

    # # Classification
    # lab_train = data.labels_binary[ind_tr]
    # lab_val = data.labels_binary[ind_val]
    # lab_test = data.labels_binary[ind_test]
    # new_output = np.concatenate([lab_train, lab_val], axis=0)
    # lab_test = lab_test.astype('float')
    # new_output = new_output.astype('float')

    # clf_rf = RandomForestClassifier(n_estimators=1000, min_impurity_decrease=0, class_weight="balanced")
    # clf_rf.fit(new_input, new_output)
    # pred_clf_rf = clf_rf.predict_proba(input_test)[:, 1]

    # fpr_roc, tpr_roc, thresholds_roc = roc_curve(lab_test, pred_clf_rf, pos_label=1)
    # area_under_roc = auc(fpr_roc, tpr_roc)
    # prec, recall, thresholds_pr = precision_recall_curve(lab_test, pred_clf_rf, pos_label=1)
    # area_under_prec = auc(recall, prec)
    # print("AUC-ROC: {:.3f}".format(area_under_roc))
    # print("AUC-PREC: {:.3f}".format(area_under_prec))