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

def analyze_good_bad_examples():
    import copy
    
    fldr = "./results/binary_classification/longitudinal_data/random_forest/selecting_good_samples/analysis_of_good_samples/rotterdam"
    dataset_name = 'Rotterdam'
    number_of_required_vfs = 3

    data = get_longitudinal_data(dataset_name, number_of_required_vfs=number_of_required_vfs, use_all_samples=True)
    data_good = copy.copy(data)
    data_bad = copy.copy(data)

    with np.load("good_indices_{:d}_vfs_{}.npz".format(number_of_required_vfs, dataset_name.lower())) as ind_file:
        good_ind_pos = ind_file['good_ind_pos']
        good_ind_neg = ind_file['good_ind_neg']

    good_ind = np.concatenate([good_ind_neg, good_ind_pos])
    bad_ind = np.setdiff1d(np.arange(data.vf.shape[0]), good_ind)   
    data_good.slice_arrays(good_ind)
    data_bad.slice_arrays(bad_ind)

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     ax.hist(d_obj.md_next, 50, density=True)
    #     ax.set_xlabel("MD of the next exams")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_next_mds.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     ax.hist(d_obj.md_curr.flatten(), 50, density=True)
    #     ax.set_xlabel(" MD (3) of the current exams")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_current_3_mds.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     ax.hist(d_obj.md_curr[:, -1], 50, density=True)
    #     ax.set_xlabel("MD of the current exams")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_current_mds.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     md_diffs = np.diff(np.concatenate([d_obj.md_curr, d_obj.md_next[:, None]], axis=1)).flatten()
    #     ax.hist(md_diffs, 50, density=True)
    #     ax.set_xlabel("MD differences")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_md_diffs.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     time_diffs = np.diff(np.concatenate([d_obj.age_curr, d_obj.age_next[:, None]], axis=1)).flatten()/30
    #     ax.hist(time_diffs, 50, density=True)
    #     ax.set_xlabel("Time differences (month)")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_time_diffs.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     mds = d_obj.md_next[d_obj.labels_binary]
    #     ax.hist(mds, 50, density=True)
    #     ax.set_xlabel("Next MDs of positive samples")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.grid()
    #     ax.set_title(title)
    # plt.savefig("{}/distributions_next_mds_positive_class.pdf".format(fldr), bbox_inches='tight')
    # plt.close()
    
    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     mds = d_obj.md_next[~d_obj.labels_binary]
    #     ax.hist(mds, 50, density=True)
    #     ax.set_xlabel("Next MDs of negative samples")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.grid()
    #     ax.set_title(title)
    # plt.savefig("{}/distributions_next_mds_negative_class.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     md_diffs = d_obj.md_curr[:, -1] - d_obj.md_next
    #     md_diffs = md_diffs[d_obj.labels_binary]
    #     ax.hist(md_diffs, 50, density=True)
    #     ax.set_xlabel("MD difference (positives)")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.grid()
    #     ax.set_title(title)
    # plt.savefig("{}/distributions_md_diffs_positive.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    # for ax, d_obj, title in zip(axes, [data, data_good, data_bad], ["all", "good", "bad"]):
    #     md_diffs = d_obj.md_curr[:, -1] - d_obj.md_next
    #     md_diffs = md_diffs[~d_obj.labels_binary]
    #     ax.hist(md_diffs, 50, density=True)
    #     ax.set_xlabel("MD difference (negatives)")
    #     ax.set_ylabel("Counts (mormalized)")
    #     ax.set_title(title)
    #     ax.grid()
    # plt.savefig("{}/distributions_md_diffs_negative.pdf".format(fldr), bbox_inches='tight')
    # plt.close()

    pid_indices = np.unique(data.pid)
    pid_train, pid_test = train_test_split(pid_indices, test_size=0.15, random_state=42)

    indices_train = np.where(np.isin(data_good.pid, pid_train))[0]
    indices_test = np.where(np.isin(data_bad.pid, pid_test))[0]

    N, _, num_locs = data_good.td.shape
    data_good.td = data_good.td.reshape(N, number_of_required_vfs * num_locs)
    N, _, num_locs = data_bad.td.shape
    data_bad.td = data_bad.td.reshape(N, number_of_required_vfs * num_locs)
 
    x_train = data_good.td[indices_train]
    x_test = data_bad.td[indices_test]
    y_train = data_good.labels_binary[indices_train]
    y_test = data_bad.labels_binary[indices_test]

    time_diffs_tr = data_good.age_next[indices_train] - data_good.age_curr[indices_train, -1]
    time_diffs_te = data_bad.age_next[indices_test] - data_good.age_curr[indices_test, -1]

    x_train = np.concatenate([x_train, data_good.age_curr[indices_train], time_diffs_tr[:, None]], axis=1)
    x_test = np.concatenate([x_test, data_bad.age_curr[indices_test], time_diffs_te[:, None]], axis=1)

    auc_roc, auc_prec, probs = run_random_forest(x_train, y_train, x_test, y_test)


def prepare_regression_data(data, number_of_history_vfs):

    # Split based on the patients
    pid_indices = np.unique(data.pid)
    pid_train, pid_test = train_test_split(pid_indices, test_size=0.15, random_state=42)
    # pid_train, pid_val = train_test_split(pid_train, test_size=0.15, random_state=30)
    
    indices_train = np.where(np.isin(data.pid, pid_train))[0]
    # indices_val = np.where(np.isin(data.pid, pid_val))[0]
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
    # x_val = data.td[indices_val]
    x_test = data.td[indices_test]

    # targets
    y_train = data.labels_binary[indices_train]
    # y_val = data.labels_binary[indices_val]
    y_test = data.labels_binary[indices_test]

    # Add current age and time difference to the input
    if number_of_history_vfs == 1:
        data.age_curr = data.age_curr[:, None]
        data.md_curr = data.md_curr[:, None]
    time_diffs = data.age_next - data.age_curr[:, -1]
    x_train = np.concatenate([x_train, data.age_curr[indices_train], time_diffs[indices_train, None]], axis=1)
    # x_val = np.concatenate([x_val, data.age_curr[indices_val], time_diffs[indices_val, None]], axis=1)
    x_test = np.concatenate([x_test, data.age_curr[indices_test], time_diffs[indices_test, None]], axis=1)

    x_train = np.concatenate([x_train, data.md_curr[indices_train]], axis=1)
    # x_val = np.concatenate([x_val, data.md_curr[indices_val]], axis=1)
    x_test = np.concatenate([x_test, data.md_curr[indices_test]], axis=1)

    # scaler = StandardScaler() # MinMaxScaler(feature_range=[-1, 1])
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test, indices_train, indices_test

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

    return area_under_roc, area_under_prec, pred_clf_rf

def identify_good_cases(dataset_name):

    # DEFS
    # dataset_name = 'Rotterdam'
    number_of_required_vfs = 3
    data = get_longitudinal_data(dataset_name, number_of_required_vfs=number_of_required_vfs, use_all_samples=True)

    if number_of_required_vfs > 1:
        N, _, num_locs = data.td.shape
        data.td = data.td.reshape(N, number_of_required_vfs * num_locs)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    
    # Split based on the patients
    pid_indices = np.unique(data.pid)

    good_ind_pos = []
    good_ind_neg = []
    bad_ind = []
    for tr_idx, test_idx in kf.split(pid_indices):
        
        pid_train = pid_indices[tr_idx]
        pid_test = pid_indices[test_idx]

        indices_train = np.where(np.isin(data.pid, pid_train))[0]
        indices_test = np.where(np.isin(data.pid, pid_test))[0]

        x_train = data.td[indices_train]
        x_test = data.td[indices_test]
        y_train = data.labels_binary[indices_train]
        y_test = data.labels_binary[indices_test]

        # Add current age and time difference to the input
        time_diffs = data.age_next - data.age_curr[:, -1]
        x_train = np.concatenate([x_train, data.md_curr[indices_train], data.age_curr[indices_train], time_diffs[indices_train, None]], axis=1)
        x_test = np.concatenate([x_test, data.md_curr[indices_test], data.age_curr[indices_test], time_diffs[indices_test, None]], axis=1)

        clf_rf = RandomForestClassifier(n_estimators=1000, min_impurity_decrease=0, class_weight="balanced")
        clf_rf.fit(x_train, y_train)
        pred_scores = clf_rf.predict_proba(x_test)[:, 1]

        # th_pos_score = np.median(pred_scores[y_test])
        # th_neg_score = np.median(pred_scores[~y_test])
        th_pos_score = np.quantile(pred_scores[y_test], 0.4)
        th_neg_score = np.quantile(pred_scores[~y_test], 0.7)

        pos_arr = indices_test[(pred_scores > th_pos_score) & (y_test)]
        neg_arr = indices_test[(pred_scores < th_neg_score) & (~y_test)]

        good_ind_pos = good_ind_pos + pos_arr.tolist()
        good_ind_neg = good_ind_neg + neg_arr.tolist()
        bad_ind = bad_ind + np.setdiff1d(indices_test, good_ind_pos + good_ind_neg).tolist()

    good_ind = good_ind_neg + good_ind_pos

    np.savez("good_indices_{:d}_vfs_{}_relaxed.npz".format(number_of_required_vfs, dataset_name.lower()), 
                    good_ind_pos=good_ind_pos, good_ind_neg=good_ind_neg)

    return good_ind, bad_ind
        
def draw_voronoi_samples(data, fldr=None):

    # Get the dimensions
    dims = data.td.shape
    
    if len(dims) == 3:
        N, n_prev_vfs, n_locs = dims
    elif len(dims) == 2:
        N, n_locs = dims
    else:
        print("Something is wrong with dimensions")
        return -1
       
    min_val = data.td.min()
    max_val = data.td.max()
    for n in range(N):
        img_list = [generate_voronoi_images_given_image_size(data.td[n, k, :][None], data.xy) for k in range(n_prev_vfs)]
        img_list = img_list + [generate_voronoi_images_given_image_size(data.td_next[n][None], data.xy)]
        time_list = [data.age_curr[n, k]/30 for k in range(n_prev_vfs)] + [data.age_next[n]/30]

        fig_grid = plt.figure(figsize=(4*(n_prev_vfs+1), 4))
        grid = ImageGrid(fig_grid, 111, nrows_ncols=(1, n_prev_vfs+1),
                        share_all=True,
                        axes_pad=0.9,
                        cbar_location='right',
                        direction='column',
                        cbar_mode='edge',
                        cbar_size='6%', cbar_pad=1)
        
    
        for i, axis in enumerate(grid):
            fh = axis.imshow(img_list[i][0], clim=[min_val, max_val], cmap='jet')
            fh.set_clip_path(Circle((30, 30), 30, transform=axis.transData))
            axis.add_patch(Circle((45, 30), 3, fc='black'))
            axis.set_xticks(np.arange(61, step=20))
            axis.set_yticks(np.arange(61, step=20))
            axis.set_xticklabels(np.arange(-30, 31, step=20))
            axis.set_yticklabels(np.arange(30, -31, step=-20))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            cbar = axis.cax.colorbar(fh)
            
            if i == 0:
                axis.set_title('Input VF at t_{:d} = {:.1f}\n ID:{:d} Eye:{:d} MD:{:.2f}, \n t_0'.format(i, 
                time_list[i], data.pid[n], data.eyeid[n], data.md_curr[n, i]))
            elif i <= n_prev_vfs-1:
                axis.set_title('Input VF at t_{:d} = {:.1f}\n ID:{:d} Eye:{:d} MD:{:.2f}, \n + {:.1f} months'.format(i, 
                time_list[i], data.pid[n], data.eyeid[n], data.md_curr[n, i], time_list[i]-time_list[i-1]))
            elif i == n_prev_vfs:
                axis.set_title('Next VF at t_{:d} = {:.1f}\n ID:{:d} Eye:{:d} MD:{:.2f} Lab:{:d}\n + {:.1f} months'.format(i, 
                time_list[i], data.pid[n], data.eyeid[n],  data.md_next[n],  data.labels_binary[n], time_list[i]-time_list[i-1]))

            cbar.ax.set_title('TD values [dB]', fontsize=7)
        
        if fldr is not None:
            plt.savefig('{}/vor_fig_{:d}.pdf'.format(fldr, n+1), bbox_inches='tight')
        plt.close()

def main():
    
    # DEFS
    dataset_name = 'Rotterdam'
    # number_of_required_vfs = 5
    number_of_required_vfs_range = [3]

    aucs_roc = []
    aucs_prec = []
    for number_of_required_vfs in number_of_required_vfs_range:
        # Get the data
        data = get_longitudinal_data(dataset_name, number_of_required_vfs=number_of_required_vfs, use_all_samples=True)

        identify_good_cases(dataset_name)

        # Work on a subset?
        with np.load("good_indices_{:d}_vfs_{}.npz".format(number_of_required_vfs, dataset_name.lower())) as ind_file:
            good_ind_pos = ind_file['good_ind_pos']
            good_ind_neg = ind_file['good_ind_neg']

        good_ind = np.concatenate([good_ind_neg, good_ind_pos])
        data.slice_arrays(good_ind)

        # draw_voronoi_samples(data, fldr='./results/binary_classification/longitudinal_data/random_forest/vor_good_samples/rotterdam')

        # Split the data
        input_tr, output_tr, input_test, \
            output_test, ind_tr, ind_test = prepare_regression_data(data, number_of_history_vfs=number_of_required_vfs)

        # Get only MD and time inputs
        # input_tr = input_tr[:, -1*(2*number_of_required_vfs+1):]
        # input_val = input_val[:, -1*(2*number_of_required_vfs+1):]
        # input_test = input_test[:, -1*(2*number_of_required_vfs+1):]

        # Combine traning and validation sets
        # new_input = np.concatenate([input_tr, input_val], axis=0)
        # new_output = np.concatenate([output_tr, output_val], axis=0)

        # Print some info
        print("Data set: {}".format(dataset_name))
        print("Number of previous VFs: {:d}".format(number_of_required_vfs))
        print("Number of training samples: {:d}".format(input_tr.shape[0]))
        # print("Number of validation samples: {:d}".format(input_val.shape[0]))
        print("Number of test samples: {:d}".format(input_test.shape[0]))
        print("Number of pos./neg. samples in training:{:d}/{:d}".format(output_tr.sum(), (output_tr==0).sum()))
        print("Number of pos./neg. samples in test:{:d}/{:d}".format(output_test.sum(), (output_test==0).sum()))

        # Classification
        auc_roc, auc_prec, probs = run_random_forest(input_tr, output_tr, input_test, output_test)
        aucs_roc.append(auc_roc)
        aucs_prec.append(auc_prec)

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(output_test, probs, pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    prec, recall, thresholds_pr = precision_recall_curve(output_test, probs, pos_label=1)
    area_under_prec = auc(recall, prec)

    plt.figure()
    plt.plot(fpr_roc, tpr_roc, label='AUC-ROC={:.3f}'.format(auc_roc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.legend()
    plt.title("ROC Curve ({})".format(dataset_name))
    plt.savefig("results/binary_classification/longitudinal_data/random_forest/roc_{}.pdf".format(dataset_name))


    plt.figure()
    plt.plot(recall, prec, label='AUC-PREC={:.3f}'.format(auc_prec))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    plt.title("PR Curve ({})".format(dataset_name))
    plt.savefig("results/binary_classification/longitudinal_data/random_forest/pr_{}.pdf".format(dataset_name))

    # plt.plot(number_of_required_vfs_range, aucs_roc, label='AUC-ROC={:.3f}'.format(auc_roc))
    # plt.plot(number_of_required_vfs_range, aucs_prec, label='AP={:.3f}'.format(auc_prec))

    IPython.embed()

if __name__ == "__main__":
    # main()
    analyze_good_bad_examples()
    # identify_good_cases()


