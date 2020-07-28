import torch
import numpy as np
import os, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from plot_conf_mat import pretty_plot_confusion_matrix
import IPython


class data_generator(object):
    """
        Generates data batches for torch tensor types

    """ 

    def  __init__(self, batch_size, shuffle = True, get_indices = False, use_cuda = False):
        
        "Initilaization"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.get_indices = get_indices
        self.use_cuda = use_cuda

    def generate(self, inputs, targets, balanced=False):
        """ Generates batches of samples

        :inputs: inputs (first dimension is number of samples)
        :targets: labels vector or matrix
        :returns: batch of inputs and targets

        """
 
        # Assertion of the equality of the number of samples for input and target
        assert len(inputs) == len(targets)

        # Indices of samples
        N = len(inputs)
        indices = np.arange(N)

        if balanced:
            classes = torch.unique(targets)
            cl_size = []
            for cl in classes:
                cl_size.append((targets == cl).sum())
            max_cl_size = max(cl_size)
            indices_extra = []
            for k, cl in enumerate(classes):
                indices_extra.append(np.random.choice(np.where(targets == cl)[0], \
                    (max_cl_size - cl_size[k]).numpy()))
            indices_extra = np.concatenate(indices_extra)
            indices = np.concatenate([indices, indices_extra])
            
        # Shuffle if true
        if self.shuffle:
            np.random.shuffle(indices)

        # Make it tensor
        if self.use_cuda:
            indices = torch.cuda.LongTensor(indices.tolist())
        else:
            indices = torch.LongTensor(indices)
        
        # Generate batches
        if self.get_indices:
            for start_idx in range(0, N, self.batch_size):
                batch_indices = indices[start_idx: np.minimum(start_idx + self.batch_size, N)]
                yield inputs[batch_indices], targets[batch_indices], batch_indices
        else:
            for start_idx in range(0, N, self.batch_size):
                batch_indices = indices[start_idx: np.minimum(start_idx + self.batch_size, N)]
                yield inputs[batch_indices], targets[batch_indices]

class data_cls():
    """Creates data object with visual fields, labels, patient ids, and mean deviations, etc."""

    def __init__(self):
        """Initializes the created object"""
        self.vf = []
        self.vf_next = []
        self.nv = []
        self.td = []
        self.td_next = []
        self.pid = []
        self.eyeid = []
        self.labels_curr = []
        self.labels = []
        self.labels_num = []
        self.md_curr = []
        self.md_next = []

    def make_arrays(self):
        """Creates array out of the attributes of type list"""
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            setattr(self, attr, np.asarray(getattr(self, attr)).squeeze())

        return 0

def define_ae(d_in, d_out):

    model = torch.nn.Sequential(
          torch.nn.Linear(d_in, 4),
          torch.nn.ReLU(),
          torch.nn.Linear(4, 4),
          torch.nn.ReLU(),
          torch.nn.Linear(4, 4),
          torch.nn.ReLU(),
          torch.nn.Linear(4, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, d_out),
          torch.nn.LogSoftmax(dim=1)
        )

    return model

class net_class(torch.nn.Module):
    """
    This makes a network object for reconstruction operation.
    """
    def __init__(self, d_in, d_out, d_hidden):

        super(net_class, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.linear_1 = torch.nn.Linear(d_in, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_hidden)
        self.linear_3 = torch.nn.Linear(d_hidden, d_hidden)
        self.linear_4 = torch.nn.Linear(d_hidden, d_hidden)
        self.linear_5 = torch.nn.Linear(d_hidden, d_out)
        self.relu = torch.nn.ReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_):
        
        out = input_
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.linear_3(out)
        out = self.relu(out)
        out = self.linear_4(out)
        out = self.relu(out)
        out = self.linear_5(out)
        out = self.logsoftmax(out)

        return out 

def get_data(dataset):
    
    if dataset == 'Rotterdam':
        # Get data
        data_file = '/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/dynStaircase100RealizationsIPS.npz'
        vf = np.load(data_file)["vf_true"][:5108]
        pId = np.load(data_file)["pId"][:5108].flatten()
        nv = np.load(data_file)["norm_vals"][:5108]
        n_samples = len(vf)

        # Delete blind spots
        vf = np.delete(vf, [25, 34], axis=1)
        nv = np.delete(nv, [25, 34], axis=1)
        td = vf - nv

        eyeId = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/eyeId_rotterdam.csv')
        xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/p24d2.csv', skiprows=1, delimiter=',')

    elif dataset == 'Bern':

        data_file = "/home/ubelix/artorg/kucur/Workspace/Data/insel_data/insel_data_G_program_extracted_on_09_04_2020_no_nans.npz"
        vf = np.load(data_file)['ph1']
        pId = np.load(data_file)['pid']
        nv = np.load(data_file)['nv']
        td = vf - nv
        n_samples = len(vf)
        eyeId = np.load(data_file, allow_pickle=True)['eyes']
        # md = -1 * np.load(data_file)['md']

        eyeId = (eyeId == "OS ").astype('uint8')
        xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/Gpattern.csv', skiprows=1, delimiter=',')

    else:
        warnings.warn("No dataset found.")
        return -1

    # Compute mean deviation (MD)
    md = (vf - nv).mean(1)

    # Classify VFs into stages
    groups = np.array(n_samples * ['normal'], dtype='object')
    groups[(md < -2) & (md > -6)] = 'early'
    groups[(md < -6) & (md > -12)] = 'moderate'
    groups[(md < -12)] = 'advanced'

    groups_num = np.zeros(groups.shape)
    groups_num[groups == 'normal'] = 0
    groups_num[groups == 'early'] = 1
    groups_num[groups == 'moderate'] = 2
    groups_num[groups == 'advanced'] = 3
 
    # Prepare data
    data = data_cls()
    pId_u = np.unique(pId)
    for each_id in pId_u:
        for e in [0, 1]:
            idx = np.where((pId == each_id) & (eyeId == e))[0]
            for k in range(len(idx)-1):
                data.vf.append(vf[idx[k]])
                data.td.append(td[idx[k]])
                data.vf_next.append(vf[idx[k+1]])
                data.td_next.append(td[idx[k+1]])
                data.labels_curr.append(groups_num[idx[k]])
                data.labels.append(groups[idx[k+1]])
                data.labels_num.append(groups_num[idx[k+1]])
                data.pid.append(each_id)
                data.eyeid.append(e)
                data.md_curr.append(md[idx[k]])
                data.md_next.append(md[idx[k+1]])
                data.xy = xy
                
    # Make arrays from the lists
    data.make_arrays()
    return data

def split_data(data):
    
    pid_indices = np.unique(data.pid)
    pid_train, pid_test = train_test_split(pid_indices, test_size=0.2, random_state=42)    
    pid_train, pid_val = train_test_split(pid_train, test_size=0.2, random_state=30)

    indices_train = np.where(np.isin(data.pid, pid_train))[0]
    indices_val = np.where(np.isin(data.pid, pid_val))[0]
    indices_test = np.where(np.isin(data.pid, pid_test))[0]

    x_train = data.td[indices_train]
    x_val = data.td[indices_val]
    x_test = data.td[indices_test]
    y_train = data.labels_num[indices_train]
    y_val = data.labels_num[indices_val]
    y_test = data.labels_num[indices_test]

    return x_train, y_train, x_val, y_val, x_test, y_test, indices_train, indices_val, indices_test

def draw_qualitative_perf(x_curr, x_next, y_true, y_pred, classes, true_class):
    
    current_examples = []
    next_examples = []
    heights = []
    num_class = len(classes)
    N = (y_true == true_class).sum()
    for cl in range(num_class):
        current_examples.append(x_curr[(y_pred == classes[cl]) & (y_true == true_class)])
        next_examples.append(x_next[(y_pred == classes[cl]) & (y_true == true_class)])
        heights.append(current_examples[-1].shape[0]/N)

    heights = np.asarray(heights)
    heights[heights < 0.01] = 0.01
    widths = [1, 1]
    
    fig = plt.figure(constrained_layout=True)
    specs = fig.add_gridspec(ncols=2, nrows=num_class, width_ratios=widths, height_ratios=heights)
    for row in range(num_class):
        ax = fig.add_subplot(specs[row, 0])
        im = ax.imshow(current_examples[row], vmin=-40, vmax=30)
        plt.axis('off')

        ax = fig.add_subplot(specs[row, 1])
        im = ax.imshow(next_examples[row], vmin=-40, vmax=30)
        ax.set_title('True = {:d}, Pred = {:d}'.format(true_class, classes[row]))
        plt.axis('off')

    plt.colorbar(im)
    return fig
    #plt.colorbar(orientation='horizontal', fraction=.1)

def generate_voronoi_images_given_image_size(data, xy_coordinates, image_size=(61, 61)):

    """ 
    Generates voronoi images using given values for patching colors and given coordiantes for seed points.
    Here image size is fixed or given by the user.
    Output conventions according to Theano CNN implemntation (4D tensor -- (number_of_samples, image_row, image_col))
    
    """

    # Number of seed locations/points
    num_locs = xy_coordinates.shape[0]
    num_obs = data.shape[0]

    # start from 0
    x = np.zeros((num_locs, 1))
    y = np.zeros((num_locs, 1))
    x[:,0] = xy_coordinates[:,0]
    y[:,0] = xy_coordinates[:,1]
    x = x + int(image_size[0]/2)
    y = y + int(image_size[1]/2)
    voronoi_points = np.column_stack((x,y))

    # A grid of full space points including seed ones
    space_coordinates = np.mgrid[0:image_size[0], 0:image_size[1]]
    x_coord = space_coordinates[0,:].flatten() # columns
    y_coord = space_coordinates[1,:].flatten() # rows
    space_coordinates = np.vstack((x_coord,y_coord)).transpose()

    # Define an image
    img_col_size = image_size[0]
    img_row_size = image_size[1]
    img = np.zeros((img_row_size, img_col_size))

    # Fill in image
    vor_images = np.zeros((num_obs, img_row_size, img_col_size))
    for k in range(num_obs):
        value_vector = data[k,:]
        for img_col_ind, img_row_ind in space_coordinates:
            dist = (voronoi_points[:, 0] - img_col_ind)**2 + (voronoi_points[:, 1] - img_row_ind)**2
            idx = np.argmin(dist)
            img[img_row_ind, img_col_ind] = value_vector[idx]

        # Have to flip because of matrix conventions (y axis coordinates increase
        # from down to up but a matrix row indices increase from up to down)
        img = np.flipud(img)
        vor_images[k, : , :] = img

    return vor_images

def draw_voronoi_samples(inputs_curr, inputs_next, labels_curr, labels_next, predicted,
                            xy, pids, eyes, mds_curr, mds_next, fldr, probs, uncertainty=None):
    
    N = len(inputs_curr)
    stages = ['Healthy', 'Early', 'Moderate', 'Advanced']
    imgs_curr = generate_voronoi_images_given_image_size(inputs_curr, xy)
    imgs_next = generate_voronoi_images_given_image_size(inputs_next, xy)
    min_val = -40
    max_val = 30
    for k in range(N):
        fig_grid = plt.figure()
        img_list = [imgs_curr[k], imgs_next[k]]
        grid = ImageGrid(fig_grid, 111, nrows_ncols=(1, 2),
                        share_all=True,
                        axes_pad=0.9,
                        cbar_location='right', 
                        direction='column',
                        cbar_mode='edge',
                        cbar_size='6%', cbar_pad=1)

        if uncertainty is not None:
            titles= ['Current VF \nPat.#{:d}, Eye{:d}, MD={:.1f}, {}'.format(int(pids[k]), int(eyes[k]), mds_curr[k], stages[int(labels_curr[k])]),
                'Next VF (Pred: {}, uncert.:{:.2f}) \nPat.#{:d}, Eye{:d}, MD={:.1f}, {}\nH={:.2f}, E={:.2f}, \nM={:.2f}, A={:.2f}'.format(stages[int(predicted[k])], uncertainty[k], 
                int(pids[k]), int(eyes[k]), mds_next[k], stages[int(labels_next[k])], probs[k, 0], probs[k, 1], probs[k, 2], probs[k, 3])]
        else:
            titles= ['Current VF \nPat.#{:d}, Eye{:d}, MD={:.1f}, {}'.format(int(pids[k]), int(eyes[k]), mds_curr[k], stages[int(labels_curr[k])]),
                'Next VF (Pred: {}) \nPat.#{:d}, Eye{:d}, MD={:.1f}, {}'.format(stages[int(predicted[k])], int(pids[k]), int(eyes[k]), mds_next[k], stages[int(labels_next[k])])]
        
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

def main():

    # DEFS
    d_out = 4
    d_hidden = 4
    device = 'cpu'
    loss = 0
    n_epochs = 150
    dataset_name = 'Rotterdam'
    folder_name = './results/{}'.format(dataset_name)
    model_filename = '{}/model'.format(folder_name)
    
    # Create the folders
    if not os.path.exists('{}/network/'.format(folder_name)):
        os.makedirs('{}/network/all_test_examples'.format(folder_name))
        os.makedirs('{}/network/mistaken_examples'.format(folder_name))
        os.makedirs('{}/network/mistaken_examples_too_confident'.format(folder_name))

    if not os.path.exists('{}/random_forest/'.format(folder_name)):
        os.makedirs('{}/random_forest/all_test_examples'.format(folder_name))
        os.makedirs('{}/random_forest/mistaken_examples'.format(folder_name))

    if not os.path.exists('{}/SVM/'.format(folder_name)):
        os.makedirs('{}/SVM/all_test_examples'.format(folder_name))
        os.makedirs('{}/SVM/mistaken_examples'.format(folder_name))

    # Get the data
    data = get_data(dataset_name)
    d_in = data.vf.shape[1]

    # Split the data
    input_tr, output_tr, input_val, output_val, input_test, \
        output_test, ind_tr, ind_val, ind_test = split_data(data)

    # Define the device
    device = torch.device(device)

    # Get the empty net
    # model = net_class(d_in, d_out, d_hidden)
    model = define_ae(d_in, d_out)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
  
    # Mini-batch generator 
    data_gen = data_generator(32)

    # Move to torch
    input_tr = torch.from_numpy(input_tr).type(torch.FloatTensor).to(device)
    output_tr = torch.from_numpy(output_tr).type(torch.long).to(device)
    input_val = torch.from_numpy(input_val).type(torch.FloatTensor).to(device)
    output_val = torch.from_numpy(output_val).type(torch.long).to(device)
    input_test = torch.from_numpy(input_test).type(torch.FloatTensor).to(device)
    output_test = torch.from_numpy(output_test).type(torch.long).to(device)

    # Loss function
    loss_fn = torch.nn.NLLLoss(reduction='sum')

    N = input_tr.size()[0]
    N_val = input_val.size()[0]
    N_test = input_test.size()[0]
    tr_loss_list= []
    val_loss_list = []
    best_val_loss = 1e6
    for epoch in range(n_epochs):

        for input_batch, output_batch in data_gen.generate(input_tr, output_tr, balanced=False):

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

        with torch.no_grad():

            # Full-pass over validation set
            pred_val = model(input_val)
            val_loss = loss_fn(pred_val, output_val)
            val_loss /= N_val
            print('Training - epoch {:d}: \t Tr. Loss: {:.5f}\t Val. Loss: {:.5f}'.format(epoch, loss.item(), val_loss.item()))
            
            # Loss history
            tr_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Keep the best model
            if model_filename != None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.clone()
                    torch.save(model.state_dict(), model_filename)

                   # Change learning rate
        scheduler.step()

    # Test
    # model_best = net_class(d_in, d_out, d_hidden)
    model_best = define_ae(d_in, d_out)
    model_best = model_best.to(device)
    model_best.load_state_dict(torch.load(model_filename, map_location=device))
    pred_test = model_best(input_test)
    pred_labels = torch.argmax(pred_test, dim=1)
    acc = (pred_labels == output_test).sum().item()/N_test
    conf_mat_net = confusion_matrix(output_test.data.numpy(), pred_labels.data.numpy())
    print("Accuracy for network: {:.3f}".format(acc))
    print(conf_mat_net)

    df_net = pd.DataFrame(conf_mat_net, columns=['Healthy', 'Early', 'Moderate', 'Advanced'], \
        index=['Healthy', 'Early', 'Moderate', 'Advanced'])

    plt = pretty_plot_confusion_matrix(df_net, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8, 8], show_null_values=1, pred_val_axis='y')
    plt.title('Confusion matrix for the network ({})'.format(dataset_name))
    plt.savefig('{}/network/confusion_matrix.pdf'.format(folder_name))
    # fig = draw_qualitative_perf(input_test.data.numpy(), data.vf_next[ind_test],\
    #      output_test.data.numpy(), pred_labels.data.numpy(), [1, 2], 2)

    # plt.show()

    # Compute entropy -- for uncertainty
    probs = np.exp(pred_test.data.numpy())
    entropy = -1 * np.sum(probs * np.log(probs), axis=1)

    acc_pred = (pred_labels == output_test).data.numpy()
    err_pred = ~acc_pred
    too_conf_inds = (err_pred) & (entropy < 0.4)
    mistakend_ind_too_confident = ind_test[(err_pred) & (entropy < 0.4)]

    # Mistaken examples
    inds = pred_labels != output_test
    mistaken_ind = ind_test[inds]
    entr_at_mistakes = entropy[pred_labels != output_test]
    if dataset_name == 'Rotterdam':
        data.vf = np.insert(data.vf, 25, 0, 1)
        data.vf = np.insert(data.vf, 34, 0, 1)
        data.vf_next = np.insert(data.vf_next, 25, 0, 1)
        data.vf_next = np.insert(data.vf_next, 34, 0, 1)
        data.td = np.insert(data.td, 25, 0, 1)
        data.td = np.insert(data.td, 34, 0, 1)
        data.td_next = np.insert(data.td_next, 25, 0, 1)
        data.td_next = np.insert(data.td_next, 34, 0, 1)

    draw_voronoi_samples(data.td[mistaken_ind], data.td_next[mistaken_ind],
        data.labels_curr[mistaken_ind], data.labels_num[mistaken_ind], pred_labels[inds], data.xy,
        data.pid[mistaken_ind], data.eyeid[mistaken_ind], data.md_curr[mistaken_ind],
        data.md_next[mistaken_ind], '{}/network/mistaken_examples/'.format(folder_name), probs[inds],
        uncertainty=entr_at_mistakes)
    
    entr_at_too_conf_mistakes = entropy[(err_pred) & (entropy < 0.4)]
    draw_voronoi_samples(data.td[mistakend_ind_too_confident], data.td_next[mistakend_ind_too_confident],
        data.labels_curr[mistakend_ind_too_confident], data.labels_num[mistakend_ind_too_confident], pred_labels[too_conf_inds], data.xy,
        data.pid[mistakend_ind_too_confident], data.eyeid[mistakend_ind_too_confident], data.md_curr[mistakend_ind_too_confident],
        data.md_next[mistakend_ind_too_confident], '{}/network/mistaken_examples_too_confident/'.format(folder_name), probs[too_conf_inds],
        uncertainty=entr_at_too_conf_mistakes)

    draw_voronoi_samples(data.td[ind_test], data.td_next[ind_test],
        data.labels_curr[ind_test], data.labels_num[ind_test], pred_labels, data.xy,
        data.pid[ind_test], data.eyeid[ind_test], data.md_curr[ind_test],
        data.md_next[ind_test], '{}/network/mistaken_examples_too_confident/'.format(folder_name), probs,
        uncertainty=entropy)

    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=100, \
        class_weight="balanced", random_state=32)
    clf_rf.fit(input_tr.data.numpy(), output_tr.data.numpy())
    pred_rf = clf_rf.predict(input_test.data.numpy())
    probs_rf = clf_rf.predict_proba(input_test.data.numpy()) + 1e-10 # Added small number to avoid numerical issues
    acc_rf = (pred_rf == output_test.data.numpy()).sum().item()/N_test
    conf_mat_rf = confusion_matrix(output_test.data.numpy(), pred_rf)
    print("Accuracy for Random Forest: {:.3f}".format(acc_rf))
    print(conf_mat_rf)

    df_rf = pd.DataFrame(conf_mat_rf, columns=['Healthy', 'Early', 'Moderate', 'Advanced'], \
        index=['Healthy', 'Early', 'Moderate', 'Advanced'])

    plt = pretty_plot_confusion_matrix(df_rf, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8, 8], show_null_values=1, pred_val_axis='y')
    plt.title('Confusion matrix for the random forest ({})'.format(dataset_name))
    plt.savefig('{}/random_forest/confusion_matrix.pdf'.format(folder_name))

    entr_rf = -1 * np.sum(probs_rf * np.log(probs_rf), axis=1)
    idx = pred_rf != output_test.data.numpy()
    mistaken_ind_rf = ind_test[idx]
    draw_voronoi_samples(data.td[mistaken_ind_rf], data.td_next[mistaken_ind_rf],
        data.labels_curr[mistaken_ind_rf], data.labels_num[mistaken_ind_rf], np.squeeze(pred_rf[idx]), 
        data.xy, data.pid[mistaken_ind_rf], data.eyeid[mistaken_ind_rf], data.md_curr[mistaken_ind_rf],
        data.md_next[mistaken_ind_rf], '{}/random_forest/mistaken_examples/'.format(folder_name), probs_rf[idx], uncertainty=entr_rf[idx])

    draw_voronoi_samples(data.td[ind_test], data.td_next[ind_test],
        data.labels_curr[ind_test], data.labels_num[ind_test], np.squeeze(pred_rf), 
        data.xy, data.pid[ind_test], data.eyeid[ind_test], data.md_curr[ind_test],
        data.md_next[ind_test], '{}/random_forest/all_test_examples/'.format(folder_name), probs_rf, uncertainty=entr_rf)

    # SVM
    clf_svm = SVC(kernel='poly', degree=1, class_weight='balanced', gamma='auto', C=0.01, probability=True)
    clf_svm.fit(input_tr.data.numpy(), output_tr.data.numpy())
    pred_svm = clf_svm.predict(input_test.data.numpy())
    probs_svm = clf_svm.predict_proba(input_test.data.numpy()) + 1e-10 # Added small number to avoid numerical issues
    acc_svm = (pred_svm == output_test.data.numpy()).sum().item()/N_test
    conf_mat_svm = confusion_matrix(output_test.data.numpy(), pred_svm)
    print("Accuracy for SVM: {:.3f}".format(acc_svm))
    print(conf_mat_svm)

    df_svm = pd.DataFrame(conf_mat_svm, columns=['Healthy', 'Early', 'Moderate', 'Advanced'], \
        index=['Healthy', 'Early', 'Moderate', 'Advanced'])

    plt = pretty_plot_confusion_matrix(df_svm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=1, pred_val_axis='y')
    plt.title('Confusion matrix for the SVM ({})'.format(dataset_name))
    plt.savefig('{}/SVM/confusion_matrix.pdf'.format(folder_name))

    entr_svm = -1 * np.sum(probs_svm * np.log(probs_svm), axis=1)
    idx = pred_svm != output_test.data.numpy()
    mistaken_ind_svm = ind_test[pred_svm != output_test.data.numpy()]
    draw_voronoi_samples(data.td[mistaken_ind_svm], data.td_next[mistaken_ind_svm],
        data.labels_curr[mistaken_ind_svm], data.labels_num[mistaken_ind_svm], np.squeeze(pred_svm[idx]),
        data.xy, data.pid[mistaken_ind_svm], data.eyeid[mistaken_ind_svm], data.md_curr[mistaken_ind_svm],
        data.md_next[mistaken_ind_svm], '{}/SVM/mistaken_examples/'.format(folder_name), probs_svm[idx], uncertainty=entr_svm[idx])

    draw_voronoi_samples(data.td[ind_test], data.td_next[ind_test],
        data.labels_curr[ind_test], data.labels_num[ind_test], np.squeeze(pred_svm),
        data.xy, data.pid[ind_test], data.eyeid[ind_test], data.md_curr[ind_test],
        data.md_next[ind_test], '{}/SVM/all_test_examples/'.format(folder_name), probs_svm, uncertainty=entr_svm)

    IPython.embed()

if __name__ == "__main__":
    main()
