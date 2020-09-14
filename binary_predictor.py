import torch
import numpy as np
import os, warnings
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.autograd import Variable
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from plot_conf_mat import pretty_plot_confusion_matrix, plot_confusion_matrix_from_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import models
from utils import *
import IPython

def main():

    # DEFS
    input_type = "voronoi"
    d_out = 1
    d_hidden = 256 # for fully connected network
    device = 'cuda'
    loss = 0
    n_epochs = 50
    batch_size = 32
    learning_rate = 1e-4
    cl_w = 5
    sch_step_size = 20
    sch_gamma = 0.1
    balanced_batch = True
    scheduler_name = "stepLR"
    dataset_name = 'Rotterdam'
    folder_name = './results/binary_classification/linear_vf/{}/network'.format(dataset_name)
    summary_folder = "{}/logs".format(folder_name)
    model_filename = '{}/model'.format(folder_name)
    data_filename = "./data_{}.pkl".format(dataset_name.lower())
    run_foldername = "runs/binary/{}/{}/{}-{}".format(dataset_name, input_type,
                        datetime.date(datetime.now()), 
                        datetime.time(datetime.now()).strftime("%H-%M"))

    summary_dict = {"n_epochs": n_epochs, "batch_size": batch_size,
                    "lr": learning_rate, "class_weights": cl_w,
                    "scheduler": scheduler_name, "scheduler_step": sch_step_size,
                    "scheduler_gamma": sch_gamma, "dataset": dataset_name, "balanced_batch": balanced_batch}

    # Create folders if not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)

    # Get the data
    # data = get_data(dataset_name, use_all_samples=True)
    data = load_object(data_filename)
    d_in = data.vf.shape[1] + 2 # Age + time diffs

    # Split the data
    input_tr, output_tr, input_val, output_val, input_test, \
        output_test, ind_tr, ind_val, ind_test = prepare_data(data, vf_format=input_type)
    
    # Define the device
    device = torch.device(device)

    # Get the empty net
    if input_type == "straight":
        # model = net_class(d_in, d_out, d_hidden)
        model = define_ae(d_in, d_out, d_hidden=d_hidden)
    elif input_type == "voronoi":
        model = simple_conv_net()
        # model = models.resnet18(pretrained=True)

    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch_step_size, gamma=sch_gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step_size, gamma=sch_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=20)

    # Mini-batch generator
    data_gen = data_generator(batch_size, get_indices=True)

    # Get the writer
    writer = SummaryWriter(run_foldername)

    # Move to torch
    input_tr = torch.from_numpy(input_tr).type(torch.FloatTensor).to(device)
    output_tr = torch.from_numpy(output_tr).type(torch.FloatTensor).to(device)[:, None]
    input_val = torch.from_numpy(input_val).type(torch.FloatTensor).to(device)
    output_val = torch.from_numpy(output_val).type(torch.FloatTensor).to(device)[:, None]
    input_test = torch.from_numpy(input_test).type(torch.FloatTensor).to(device)
    output_test = torch.from_numpy(output_test).type(torch.FloatTensor).to(device)[:, None]

    # Number of samples
    N = input_tr.shape[0]
    N_val = input_val.shape[0]
    N_test = input_test.shape[0]

    # Loss function
    # pos_labes_perc = (output_test == 1).sum().item()/N_test
    # class_weights = torch.FloatTensor([pos_labes_perc, 1-pos_labes_perc])
    pos_weight = torch.FloatTensor([cl_w]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    tr_loss_list= []
    val_loss_list = []
    best_val_loss = 1e6
    for epoch in range(n_epochs):

        for input_batch, output_batch, _ in data_gen.generate(input_tr, output_tr, balanced=balanced_batch):

            # Feed-forward
            pred = model(input_batch)

            # Current loss and cumulative loss
            loss_curr = loss_fn(pred, output_batch)
            loss += loss_curr
            loss_curr = loss_curr / input_batch.size()[1]
            
            # Zero the gradients, backpropagate the gradients, update the params
            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
        
        # Average training loss over samples
        loss /= N
        writer.add_scalar("Loss/train", loss, epoch)

        with torch.no_grad():

            # Full-pass over validation set
            pred_val = model(input_val)
            val_loss = loss_fn(pred_val, output_val)
            val_loss /= N_val
            writer.add_scalar("Loss/validation", val_loss, epoch)
            print('Training - epoch {:d}: \t Tr. Loss: {:.5f}\t Val. Loss: {:.5f}'.format(epoch, loss.item(), val_loss.item()))
            
            # Loss history
            tr_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Keep the best model
            if model_filename is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.clone()
                    torch.save(model.state_dict(), model_filename)

        # Change learning rate
        scheduler.step()
        # scheduler.step(val_loss)

    # Test
    if input_type == "straight":
        # model_best = net_class(d_in, d_out, d_hidden)
        model_best = define_ae(d_in, d_out, d_hidden=d_hidden)
    elif input_type == "voronoi":
        model_best = simple_conv_net()
    model_best = model_best.to(device)
    model_best.load_state_dict(torch.load(model_filename, map_location=device))
    pred_test = model_best(input_test)

    # Write best model into one of the run files
    torch.save(model.state_dict(), "{}/model".format(run_foldername))

    # ROC curve for the network and select the best cut-off value
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(output_test.cpu().data.numpy(), pred_test.cpu().data.numpy(), pos_label=1)
    area_under_roc = auc(fpr_roc, tpr_roc)
    opt_th_roc = thresholds_roc[np.argmax(tpr_roc + (1-fpr_roc))]
    pred_labels = (pred_test > opt_th_roc).type(torch.FloatTensor).to(device)
    acc = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net_roc = confusion_matrix(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    f1_score_net = f1_score(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    tnr = conf_mat_net_roc[0, 0]/conf_mat_net_roc[0].sum()
    tpr = conf_mat_net_roc[1, 1]/conf_mat_net_roc[1].sum()
    balanced_acc_net = 0.5*(tnr+tpr)
    print(classification_report(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy()))
    print("Area under ROC curve: {:.3f}".format(area_under_roc))
    print("Accuracy for network: {:.3f}".format(acc))
    print("Balanced accuracy for network: {:.3f}".format(balanced_acc_net))
    print("F1-score for netwrok: {:.3f}".format(f1_score_net))
    print(conf_mat_net_roc)
    plt, fig = plot_confusion_matrix_from_data(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy(), \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (ROC) for the network ({})'.format(dataset_name))
    plt.savefig('{}/confusion_matrix_roc.pdf'.format(folder_name))
    writer.add_figure("Confusion matrix/ROC/", fig)
    writer.add_scalar("Test/AUC-ROC", area_under_roc)
    writer.add_text("AUC/ROC", str(area_under_roc))

    
    # PREC curve for the network and select the best cut-off valu
    prec, recall, thresholds_pr = precision_recall_curve(output_test.cpu().data.numpy(),\
             pred_test.cpu().data.numpy(), pos_label=1)
    opt_th_pr = thresholds_pr[np.argmax(prec * recall/(prec+recall+1e-10))]
    area_under_prec = auc(recall, prec)
    pred_labels = (pred_test > opt_th_pr).type(torch.FloatTensor).to(device)
    acc = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net_pr = confusion_matrix(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    f1_score_net = f1_score(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    tnr = conf_mat_net_pr[0, 0]/conf_mat_net_pr[0].sum()
    tpr = conf_mat_net_pr[1, 1]/conf_mat_net_pr[1].sum()
    balanced_acc_net = 0.5*(tnr+tpr)
    print(classification_report(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy()))
    print("Area under PR curve: {:.3f}".format(area_under_prec))
    print("Accuracy for network: {:.3f}".format(acc))
    print("Balanced accuracy for network: {:.3f}".format(balanced_acc_net))
    print("F1-score for netwrok: {:.3f}".format(f1_score_net))
    print(conf_mat_net_pr)
    plt, fig = plot_confusion_matrix_from_data(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy(), \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix (PR) for the network ({})'.format(dataset_name))
    plt.savefig('{}/confusion_matrix_pr.pdf'.format(folder_name))
    writer.add_figure("Confusion matrix/PR/", fig)
    writer.add_scalar("Test/AUC-PR", area_under_prec)
    writer.add_text("AUC/PR", str(area_under_prec))

    # Write training parameters to tensorboard
    param_str = ""
    for key, value in summary_dict.items():
        temp = key + ": " + str(value) + "  \n"
        param_str += temp
    writer.add_text("Network/training-params", param_str)
    writer.add_text("Network/model", str(model))
    writer.flush()

    # Baseline
    pred_base = np.zeros((N_test, 1))
    acc_base = (pred_base == output_test.cpu().data.numpy()).sum()/N_test
    conf_mat_base = confusion_matrix(output_test.cpu().data.numpy(), pred_base)
    f1_score_base = f1_score(output_test.cpu().data.numpy(), pred_base)
    tnr = conf_mat_base[0, 0]/conf_mat_base[0].sum()
    tpr = conf_mat_base[1, 1]/conf_mat_base[1].sum()
    balanced_acc_base = 0.5*(tnr+tpr)
    print("Accuracy for baseline: {:.3f}".format(acc_base))
    print("Balanced accuracy for baseline: {:.3f}".format(balanced_acc_base))
    print("F1-score for baseline: {:.3f}".format(f1_score_base))
    # print(classification_report(output_test.data.numpy(), pred_base))
    print(conf_mat_base)
    IPython.embed()

if __name__ == "__main__":
    main()
