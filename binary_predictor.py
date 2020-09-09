import torch
import numpy as np
import os, warnings
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve
from torch.autograd import Variable
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from plot_conf_mat import pretty_plot_confusion_matrix, plot_confusion_matrix_from_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from torch.utils.tensorboard import SummaryWriter
from utils import *
import IPython

def main():

    # DEFS
    input_type = "voronoi"
    d_out = 1
    d_hidden = 2048
    device = 'cuda'
    loss = 0
    n_epochs = 20
    batch_size = 32
    learning_rate = 0.0001
    cl_w = [1, 1]
    sch_step_size = 10
    sch_gamma = 0.7
    scheduler_name = "stepLR"
    dataset_name = 'Rotterdam'
    folder_name = './results/binary/trial/{}'.format(dataset_name)
    model_filename = '{}/model'.format(folder_name)
    data_filename = "./data_{}.pkl".format(dataset_name.lower())

    summary_dict = {"n_epochs": n_epochs, "batch_size": batch_size,
                    "lr": learning_rate, "class_weights": cl_w,
                    "scheduler": scheduler_name, "scheduler_step": sch_step_size,
                    "scheduler_gamma": sch_gamma, "dataset": dataset_name}

    if not os.path.exists(folder_name):
        os.makedirs('{}/network/all_test_examples'.format(folder_name))
    
    # # Create the folders
    # if not os.path.exists('{}/network/'.format(folder_name)):
    #     os.makedirs('{}/network/all_test_examples'.format(folder_name))
    #     # os.makedirs('{}/network/progressed_test_examples'.format(folder_name))
    #     os.makedirs('{}/network/mistaken_examples'.format(folder_name))
    #     os.makedirs('{}/network/mistaken_examples_too_confident'.format(folder_name))

    # if not os.path.exists('{}/random_forest/'.format(folder_name)):
    #     os.makedirs('{}/random_forest/all_test_examples'.format(folder_name))
    #     os.makedirs('{}/random_forest/mistaken_examples'.format(folder_name))

    # if not os.path.exists('{}/SVM/'.format(folder_name)):
    #     os.makedirs('{}/SVM/all_test_examples'.format(folder_name))
    #     os.makedirs('{}/SVM/mistaken_examples'.format(folder_name))

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
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step_size, gamma=sch_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=20)

    # Mini-batch generator
    data_gen = data_generator(batch_size, get_indices=True)

    # Get the writer
    writer = SummaryWriter()

    # Move to torch
    input_tr = torch.from_numpy(input_tr).type(torch.FloatTensor).to(device)
    output_tr = torch.from_numpy(output_tr).type(torch.long).to(device)
    input_val = torch.from_numpy(input_val).type(torch.FloatTensor).to(device)
    output_val = torch.from_numpy(output_val).type(torch.long).to(device)
    input_test = torch.from_numpy(input_test).type(torch.FloatTensor).to(device)
    output_test = torch.from_numpy(output_test).type(torch.long).to(device)

    # Number of samples
    N = input_tr.shape[0]
    N_val = input_val.shape[0]
    N_test = input_test.shape[0]

    # Loss function
    # pos_labes_perc = (output_test == 1).sum().item()/N_test
    # class_weights = torch.FloatTensor([pos_labes_perc, 1-pos_labes_perc])
    class_weights = torch.FloatTensor(cl_w).to(device)
    loss_fn = torch.nn.NLLLoss(reduction='sum', weight=class_weights)

    tr_loss_list= []
    val_loss_list = []
    best_val_loss = 1e6
    for epoch in range(n_epochs):

        for input_batch, output_batch, _ in data_gen.generate(input_tr, output_tr, balanced=True):

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

    # ROC curve for the network and select the best cut-off value
    fpr_roc, tnr_roc, thresholds = roc_curve(output_test.cpu().data.numpy(), pred_test.cpu().data.numpy(), pos_label=1)
    opt_th = thresholds[np.argmax(tnr_roc + (1-fpr_roc))]

    # Calculate metrics for the cut-off value and print them
    pred_labels = pred_test > opt_th
    acc = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net = confusion_matrix(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    f1_score_net = f1_score(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy())
    tnr = conf_mat_net[0, 0]/conf_mat_net[0].sum()
    tpr = conf_mat_net[1, 1]/conf_mat_net[1].sum()
    balanced_acc_net = 0.5*(tnr+tpr)
    print(classification_report(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy()))
    print("Accuracy for network: {:.3f}".format(acc))
    print("Balanced accuracy for network: {:.3f}".format(balanced_acc_net))
    print("F1-score for netwrok: {:.3f}".format(f1_score_net))
    print(conf_mat_net)

    plt, fig = plot_confusion_matrix_from_data(output_test.cpu().data.numpy(), pred_labels.cpu().data.numpy(), \
        show_null_values=1, pred_val_axis='col', columns=["Stable", "Progressed"])
    plt.title('Confusion matrix for the network ({})'.format(dataset_name))
    plt.savefig('{}/network/confusion_matrix.pdf'.format(folder_name))
    writer.add_figure("Confusion matrix/test", fig)

    # Write training parameters to tensorboard
    param_str = ""
    for key, value in summary_dict.items():
        temp = key + ": " + str(value) + "  \n"
        param_str += temp
    writer.add_text("Network/training-params", param_str)
    writer.add_text("Network/model", str(model))
    writer.flush()

    # Baseline
    pred_base = np.zeros((N_test))
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
