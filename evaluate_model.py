import torch
import numpy as np
import os, sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.autograd import Variable
import pandas as pd
from plot_conf_mat import pretty_plot_confusion_matrix, plot_confusion_matrix_from_data
from utils import *
import IPython


def evaluate_model(model_filename, input_test, output_test, dataset_name, input_type="straight", device='cuda', folder_name=None):

    N_test = input_test.shape[0]

    if input_type == "straight":
        d_in = input_test.shape[1] + 2
        d_out = 1
        d_hidden = 256
        model = define_ae(d_in, d_out, d_hidden=d_hidden)
    elif input_type == "voronoi":
        model = simple_conv_net()
    model = model.to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    pred_test = model(input_test)

    # ROC curve for the network and select the best cut-off value
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(output_test.cpu().data.numpy(), pred_test.cpu().data.numpy(), pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    opt_th_roc = thresholds_roc[np.argmax(tpr_roc + (1-fpr_roc))]
    pred_labels = (pred_test > opt_th_roc).type(torch.FloatTensor).to(device)
    acc_roc = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net_roc = confusion_matrix(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())

    # PR curve for the network and select the best cut-off valu
    prec, recall, thresholds_pr = precision_recall_curve(output_test.cpu().data.numpy(),\
             pred_test.cpu().data.numpy(), pos_label=1)
    opt_th_pr = thresholds_pr[np.argmax(prec * recall/(prec+recall+1e-10))]
    area_under_prec = auc(recall, prec)
    pred_labels = (pred_test > opt_th_pr).type(torch.FloatTensor).to(device)
    acc_pr = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net_pr = confusion_matrix(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())

    # print the results
    print(conf_mat_net_roc)
    print(conf_mat_net_pr)
    print("Area under ROC curve: {:.3f}".format(area_under_roc))
    print("Area under PR curve: {:.3f}".format(area_under_prec))
    # print("Accuracy (optimal threshold PR): {:.3f}".format(acc))
    # print("Accuracy (optimal threshold ROC): {:.3f}".format(acc))
    
    # Plot confusion matrices
    plt, fig = plot_confusion_matrix_from_data(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy(), \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (ROC) for the network ({})'.format(dataset_name))
    if folder_name is not None:
        plt.savefig('{}/confusion_matrix_pr.pdf'.format(folder_name))
    else:
        plt.show()

    plt, fig = plot_confusion_matrix_from_data(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy(), \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (PR) for the network ({})'.format(dataset_name))
    
    if folder_name is not None:
        plt.savefig('{}/confusion_matrix_pr.pdf'.format(folder_name))
    else:
        plt.show()

if __name__ == "__main__":
    
    filename = sys.argv[1]
    dataset_name = "Rotterdam"
    input_tyep = "voronoi"
    data_filename = "./data_{}.pkl".format(dataset_name.lower())
    data = load_object(data_filename)

    _, _, _, _, input_test, output_test, _, _, ind_test = prepare_data(data, vf_format=input_type)