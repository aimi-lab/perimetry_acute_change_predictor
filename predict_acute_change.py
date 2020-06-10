import torch
import numpy as np
import os, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import matplotlib.pyplot as plt
import IPython


class data_generator(object):
    """
        Generates data batches for torch tensor types

    """ 

    def  __init__(self,batch_size,shuffle = True, get_indices = False, use_cuda = False):
        
        "Initilaization"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.get_indices = get_indices
        self.use_cuda = use_cuda

    def generate(self,inputs,targets):
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
        self.pid = []
        self.eyeid = []
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

class net_class(torch.nn.Module):
    """
    This makes a network object for reconstruction operation.
    """
    def __init__(self, d_in, d_hidden, d_out, is_conv = False):

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
        self.logsoftmax = torch.nn.LogSoftmax()

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
    
    if dataset == 'rotterdam':
        # Get data
        data_file = '/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/dynStaircase100RealizationsIPS.npz'
        vf = np.load(data_file)["vf_true"][:5108]
        pId = np.load(data_file)["pId"][:5108].flatten()
        nv = np.load(data_file)["norm_vals"][:5108]
        n_samples = len(vf)

        # Delete blind spots
        vf = np.delete(vf, [25, 34], axis=1)
        nv = np.delete(nv, [25, 34], axis=1)

        eyeId = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/eyeId_rotterdam.csv')

    # elif self.dataset_name == 'bern':

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
                data.vf_next.append(vf[idx[k+1]])
                data.labels.append(groups[idx[k+1]])
                data.labels_num.append(groups_num[idx[k+1]])
                data.pid.append(each_id)
                data.eyeid.append(e)
                data.md_curr.append(md[idx[k]])
                data.md_next.append(md[idx[k+1]])
                
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

    x_train = data.vf[indices_train]
    x_val = data.vf[indices_val]
    x_test = data.vf[indices_test]
    y_train = data.labels_num[indices_train]
    y_val = data.labels_num[indices_val]
    y_test = data.labels_num[indices_test]   

    return x_train, y_train, x_val, y_val, x_test, y_test, indices_train, indices_val, indices_test

def draw_qualitative_perf(x_curr, x_next, y_true, y_pred, num_class, true_class):
    
    current_examples = []
    next_examples = []
    heights = []
    N = (y_true == true_class).sum()
    for cl in range(num_class):
        current_examples.append(x_curr[(y_pred == cl) & (y_true == true_class)])
        next_examples.append(x_next[(y_pred == cl) & (y_true == true_class)])
        heights.append(current_examples[-1].shape[0]/N)

    heights = np.asarray(heights)
    heights[heights < 0.01] = 0.01
    widths = [1, 1]
    
    fig = plt.figure(constrained_layout=True)
    specs = fig.add_gridspec(ncols=2, nrows=num_class, width_ratios=widths, height_ratios=heights)
    for row in range(num_class):
        ax = fig.add_subplot(specs[row, 0])
        im = ax.imshow(current_examples[row], vmin=0, vmax=40)
        plt.axis('off')

        ax = fig.add_subplot(specs[row, 1])
        im = ax.imshow(next_examples[row], vmin=0, vmax=40)
        ax.set_title('True = {:d}, Pred = {:d}'.format(true_class, row))
        plt.axis('off')

    plt.colorbar(im)
    return fig
    #plt.colorbar(orientation='horizontal', fraction=.1)

def main():

    # DEFS
    d_in = 52
    d_out = 4
    d_hidden = 512
    device = 'cpu'
    loss = 0
    n_epochs = 150
    model_filename = 'dummy'

    # Get the data
    data = get_data('rotterdam')

    # Split the data
    input_tr, output_tr, input_val, output_val, input_test, \
        output_test, ind_tr, ind_val, ind_test = split_data(data)

    # Define the device
    device = torch.device(device)

    # Get the empty net
    model = net_class(d_in, d_out, d_hidden)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        for input_batch, output_batch in data_gen.generate(input_tr, output_tr):

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
    model_best = net_class(d_in, d_out, d_hidden)
    model_best = model_best.to(device)
    model_best.load_state_dict(torch.load(model_filename, map_location=device))
    pred_test = model_best(input_test)
    pred_labels = torch.argmax(pred_test, dim=1)
    acc = (pred_labels == output_test).sum().item()/N_test
    conf_mat = confusion_matrix(output_test.data.numpy(), pred_labels.data.numpy())
    print(acc)
    print(conf_mat)

    fig = draw_qualitative_perf(input_test.data.numpy(), data.vf_next[ind_test],\
         output_test.data.numpy(), pred_labels.data.numpy(), 4, 0)

    plt.show()
    IPython.embed()

if __name__ == "__main__":
    main()
