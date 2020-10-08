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

def draw_voronoi_samples(data, predicted, probs, uncertainty, fldr):

    N = data.vf.shape[0]
    stages = ['Healthy', 'Early', 'Moderate', 'Advanced']   
    classes = ["Stable", "Progressed"]
    min_val = -40
    max_val = 30
    for k in range(N):
        fig_grid = plt.figure()
        img_list = [data.vor_images[k], data.vor_images_next[k]]
        grid = ImageGrid(fig_grid, 111, nrows_ncols=(1, 2),
                        share_all=True,
                        axes_pad=0.9,
                        cbar_location='right', 
                        direction='column',
                        cbar_mode='edge',
                        cbar_size='6%', cbar_pad=1)

        time_diff = (data.age_next[k] - data.age_curr[k])/30 # in months
        titles= ['Current VF \nPat.#{:d}, Eye{:d}, MD={:.1f}, {}'.format(int(data.pid[k]), int(data.eyeid[k]), data.md_curr[k], stages[int(data.labels_curr[k])]),
                'Next VF (in {:.1f} months)\n Prediction: {}, uncert.:{:.2f} \nPat.#{:d}, Eye{:d}, MD={:.1f}, {} \nProb.={:.2f}'.format(time_diff, classes[int(predicted[k])], uncertainty[k, 0],
                int(data.pid[k]), int(data.eyeid[k]), data.md_next[k], stages[int(data.labels_num[k])], probs[k, 0])]
        
        for i, axis in enumerate(grid):
            fh = axis.imshow(img_list[i], clim=[min_val, max_val], cmap='jet')
            fh.set_clip_path(Circle((30, 30), 30, transform=axis.transData))
            axis.add_patch(Circle((45, 30), 3, fc='black'))
            axis.set_xticks(np.arange(61, step=20))
            axis.set_yticks(np.arange(61, step=20))
            axis.set_xticklabels(np.arange(-30, 31, step=20))
            axis.set_yticklabels(np.arange(30, -31, step=-20))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            cbar = axis.cax.colorbar(fh)
            # if i < len(inputs):
            #     axis.set_title(r'Input \n $t_{N-{:d}}$')
            axis.set_title(titles[i], fontsize=7)
            cbar.ax.set_title('TD values [dB]', fontsize=7)

        plt.savefig('{}/vor_fig_{:d}.pdf'.format(fldr, k+1), bbox_inches='tight')
        plt.close()
        # print('Image {:d} is done...'.format(k+1))

def evaluate_classifier(targets, preds, probs, data, binary=True, folder_name=None):

    N_test = targets.shape[0]
    if folder_name is not None:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
   
    if binary: # assumes probs for single output
        entropy = -1 * (probs * np.log(probs) + (1-probs) * np.log(1-probs))
    else: # assumes probs for mutliclass outputs
        entropy = -1 * np.sum(probs * np.log(probs), axis=1)

    ######################################################################################################
    # ROC curve for the network and select the best cut-off value
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(targets, preds, pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    opt_th_roc = thresholds_roc[np.argmax(tpr_roc + (1-fpr_roc))]
    pred_labels = (preds > opt_th_roc)
    acc_roc = (pred_labels == targets).sum()/N_test
    conf_mat_net_roc = confusion_matrix(targets, pred_labels)

    # PR curve for the network and select the best cut-off valu
    prec, recall, thresholds_pr = precision_recall_curve(targets,\
             preds, pos_label=1)
    opt_th_pr = thresholds_pr[np.argmax(prec * recall/(prec+recall+1e-10))]
    area_under_pr = auc(recall, prec)
    pred_labels = (preds > opt_th_pr)
    acc_pr = (pred_labels == targets).sum()/N_test
    conf_mat_net_pr = confusion_matrix(targets, pred_labels)

    ######################################################################################################
    # print the results
    print(conf_mat_net_roc)
    print(conf_mat_net_pr)
    print("Area under ROC curve: {:.3f}".format(area_under_roc))
    print("Area under PR curve: {:.3f}".format(area_under_pr))
    # print("Accuracy (optimal threshold PR): {:.3f}".format(acc))
    # print("Accuracy (optimal threshold ROC): {:.3f}".format(acc))

    ######################################################################################################
    # Plot the ROC and PR curves
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr_roc, tpr_roc, label="AUC={:.3f}".format(area_under_roc))
    plt.grid()
    plt.xlabel('FPR')
    plt.xlabel('TPR')
    plt.legend()
    if folder_name is not None:
        plt.savefig('{}/roc_curve.pdf'.format(folder_name))
    plt.close()

    plt.figure()
    plt.plot(recall, prec, label="AUC={:.3f}".format(area_under_pr))
    plt.grid()
    plt.xlabel('Recall')
    plt.xlabel('Precision')
    plt.legend()
    if folder_name is not None:
        plt.savefig('{}/pr_curve.pdf'.format(folder_name))
    
    ######################################################################################################      
    # Plot confusion matrices
    plt, fig = plot_confusion_matrix_from_data(targets, pred_labels, \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (ROC) for the network ({})'.format(dataset_name))
    if folder_name is not None:
        plt.savefig('{}/confusion_matrix_pr.pdf'.format(folder_name))
    else:
        plt.show()

    plt, fig = plot_confusion_matrix_from_data(targets, pred_labels, \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (PR) for the network ({})'.format(dataset_name))
    
    if folder_name is not None:
        plt.savefig('{}/confusion_matrix_pr.pdf'.format(folder_name))
    else:
        plt.show()

    ######################################################################################################
    # Qualitative examples

    # Mistaken cases
    acc_pred = (pred_labels == targets)
    err_pred = ~acc_pred.squeeze()
    # too_conf_inds = (err_pred) & (entropy < 0.4)
    # mistakend_ind_too_confident = ind_test[(err_pred) & (entropy < 0.4)]
    data.slice_arrays(err_pred)
    if folder_name is not None:
        mistakes_folder = "{}/mistaken_examples/".format(folder_name)
        if not os.path.exists(mistakes_folder):
            os.makedirs(mistakes_folder)
        draw_voronoi_samples(data, pred_labels[err_pred], probs[err_pred], entropy[err_pred], mistakes_folder)

def evaluate_network(model_filename, data, input_test, output_test, dataset_name, 
                    input_type="straight", device='cuda', folder_name=None):

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
    pred_test = model(input_test).cpu().data.numpy()
    probs = 1/(1+np.exp(-1*pred_test)) # sigmoid

    evaluate_classifier(output_test.cpu().data.numpy(), pred_test, probs, data, folder_name=folder_name)

if __name__ == "__main__":
    
    folder_name = "./runs/binary/Bern/voronoi/2020-09-14-16-18/"
    model_filename = "./runs/binary/Bern/voronoi/2020-09-14-16-18/model" #sys.argv[1]
    dataset_name = "Bern"
    input_type = "voronoi"
    device = "cuda"
    data_filename = "./data_{}.pkl".format(dataset_name.lower())
    data = load_object(data_filename)

    _, _, _, _, input_test, output_test, _, _, ind_test = prepare_data(data, vf_format=input_type)

    # Move to torch
    input_test = torch.from_numpy(input_test).type(torch.FloatTensor).to(device)
    output_test = torch.from_numpy(output_test).type(torch.FloatTensor).to(device)[:, None]

    data.slice_arrays(ind_test) # in-place operation
    evaluate_network(model_filename, data, input_test, output_test, dataset_name,
        input_type=input_type, device=device, folder_name=folder_name)

