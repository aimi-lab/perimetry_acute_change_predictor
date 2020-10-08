import numpy as np
import torch, pickle, sys
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from plot_conf_mat import pretty_plot_confusion_matrix
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
sys.path.append('/home/ubelix/artorg/kucur/Workspace/Modules')
import utils_pytorch

# AUX

def save_object(obj, filename):
    """
    Saves an object ot a file
    Inputs:
        obj: Object to write into the file
        filename: The filename to write into
    Outputs:
        File with filename
    """        
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """
    Load the object in a pickle file into the environment
    Inputs: 
        filename: the filename were object to load was saved
    Outputs:
        obj: the object of interest, can be a list of objects
    """
    with open(filename, 'rb') as input:
        obj = pickle.load(input)

    return obj

# PREPROCESSING
def standardize_image(img, means_ch=None, stds_ch=None):
    """Standardization of image channels separately (img size = N x num_ch x height x width) """

    N, num_ch, h, w = img.shape
    img_stdized = np.zeros(img.shape)

    if means_ch == None:
            means_ch = [np.mean(img[:, ch, :, :]) for ch in range(num_ch)]
    if stds_ch == None:
            stds_ch = [np.std(img[:, ch, :, :]) for ch in range(num_ch)]
        
    for ch in range(num_ch):
        img_stdized[:, ch, :, :] = (img[:, ch, :, :] - means_ch[ch])/stds_ch[ch] 
    
    return img_stdized, means_ch, stds_ch

# DATA

class data_generator(object):
    """
        Generates data batches for torch tensor types

    """ 

    def  __init__(self, batch_size, shuffle=True, get_indices=False, use_cuda=False):
        
        "Initilaization"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.get_indices = get_indices
        self.use_cuda = use_cuda

    def generate(self, inputs, targets, balanced=False):
        """ Generates batches of samples
        for Pytorch!!!

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
                temp_ind_vector = torch.where(targets == cl)[0]
                perm = torch.randperm(temp_ind_vector.size(0))
                indices_extra.append(perm[:(max_cl_size - cl_size[k])])
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
        # self.nv = []
        self.td = []
        self.td_next = []
        self.pid = []
        self.eyeid = []
        self.labels_curr = []
        self.labels = []
        self.labels_num = []
        self.md_curr = []
        self.md_next = []
        self.age_curr = []
        self.age_next = []
        self.labels_binary = []

    def make_arrays(self):
        """Creates array out of the attributes of type list"""
        attrs = list(self.__dict__.keys())
        for attr in attrs:
            setattr(self, attr, np.asarray(getattr(self, attr)).squeeze())

    def slice_arrays(self, indices):
        """Slices the arrays with given indices"""

        attrs = list(self.__dict__.keys())
        for attr in attrs:
            if attr is not "xy":
                setattr(self, attr, getattr(self, attr)[indices])

def get_longitudinal_data(dataset, number_of_required_vfs, keep_blind_spot=False, use_all_samples=True, filename=None):
    
    if dataset == 'Rotterdam':

        # Get data
        data_file = '/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/rotterdam_numpy_data_05_08_2020.npz'
        with np.load(data_file, allow_pickle=True) as peri_data:
            vf = peri_data["vf"]
            td = peri_data["td"]
            pId = peri_data["pid"]
            nv = peri_data["nv"]
            ages = peri_data['ages']
            eyeId = peri_data['eyes']
        n_samples = len(vf)
        xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/p24d2.csv', skiprows=1, delimiter=',')
        td[np.isnan(td)] = -40
        td[td==-1000] = -40

        if keep_blind_spot is False:
            # Then delete blind spots
            vf = np.delete(vf, [25, 34], axis=1)
            nv = np.delete(nv, [25, 34], axis=1)
            td = np.delete(td, [25, 34], axis=1)
            xy = np.delete(xy, [25, 34], axis=0)

    elif dataset == 'Bern':

        data_file = "/home/ubelix/artorg/kucur/Workspace/Data/insel_data/insel_data_G_program_extracted_on_04_08_2020_no_nans.npz"
        with np.load(data_file, allow_pickle=True) as peri_data:
            vf = peri_data['ph1']
            pId = peri_data['pid']
            nv = peri_data['nv']
            ages = peri_data['ages']
            eyeId = peri_data['eyes']
            xy = peri_data['xy']
        td = vf - nv
        n_samples = len(vf)
        # md = -1 * np.load(data_file)['md']
        eyeId = (eyeId == "OS ").astype('uint8')
        # xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/Gpattern.csv', skiprows=1, delimiter=',')

    else:
        warnings.warn("No dataset found.")
        return -1

    # Compute mean deviation (MD)
    md = (vf - nv).mean(1)

    # Classify VFs into stages
    groups = np.array(n_samples * ['normal'], dtype='object')
    groups[(md <= -2) & (md > -6)] = 'early'
    groups[(md <= -6) & (md > -12)] = 'moderate'
    groups[(md <= -12)] = 'advanced'

    groups_num = np.zeros(groups.shape)
    groups_num[groups == 'normal'] = 0
    groups_num[groups == 'early'] = 1
    groups_num[groups == 'moderate'] = 2
    groups_num[groups == 'advanced'] = 3

    # Do we limit the time diffs between samples
    if use_all_samples:
        min_time_diff = 0
        max_time_diff = 1e10
    else:
        min_time_diff = 5
        max_time_diff = 24

    # Prepare data
    data = data_cls()
    pId_u = np.unique(pId)
    for each_id in pId_u:
        for e in [0, 1]:
            idx = np.where((pId == each_id) & (eyeId == e))[0]
            aux = ages[idx]
            if np.all(np.diff(aux) >= 0): # Should be sorted!
                number_of_available_vfs = len(idx)
                if (number_of_required_vfs <= (number_of_available_vfs-1)):
                    for k in range(number_of_available_vfs-number_of_required_vfs):
                        age_stack = [ages[idx[k+n]] for n in range(number_of_required_vfs)]
                        time_diff = (ages[idx[k+number_of_required_vfs]] - ages[idx[k+number_of_required_vfs-1]])/30.0
                        if np.all(np.diff(age_stack)/30.0 >= min_time_diff) and np.all(np.diff(age_stack)/30.0 <= max_time_diff) and \
                                     ((time_diff >= min_time_diff) & (time_diff <= max_time_diff)):
                            vf_stack = [vf[idx[k+n]] for n in range(number_of_required_vfs)]
                            td_stack = [td[idx[k+n]] for n in range(number_of_required_vfs)]
                            label_curr_stack = [groups_num[idx[k+n]] for n in range(number_of_required_vfs)]
                            md_stack = [md[idx[k+n]] for n in range(number_of_required_vfs)]
                            age_stack = [ages[idx[k+n]] for n in range(number_of_required_vfs)]
                            data.vf.append(np.stack(vf_stack))
                            data.td.append(np.stack(td_stack))
                            data.vf_next.append(vf[idx[k + number_of_required_vfs]])
                            data.td_next.append(td[idx[k + number_of_required_vfs]])
                            data.labels_curr.append(np.stack(label_curr_stack))
                            data.labels.append(groups[idx[k + number_of_required_vfs]])
                            data.labels_num.append(groups_num[idx[k + number_of_required_vfs]])
                            data.pid.append(each_id)
                            data.eyeid.append(e)
                            data.md_curr.append(md_stack)
                            data.md_next.append(md[idx[k + number_of_required_vfs]])
                            data.age_curr.append(age_stack)
                            data.age_next.append(ages[idx[k + number_of_required_vfs]])
                            data.labels_binary.append((groups_num[idx[k + number_of_required_vfs]] - \
                            groups_num[idx[k + number_of_required_vfs - 1]]) >= 1) # 1 if prgressed, 0 otherwise
            else:
                print("{:.0f}-{:.0f}".format(each_id, e))

    # Add the coordinates
    data.xy = xy

    # Make arrays from the lists
    data.make_arrays()

    # Add dummy dimensions

    # Check the number of samples 
    print("Number of total samples:{:d}".format(data.vf.shape[0]))
    
    # Save if you want
    if filename is not None:
        save_object(data, filename)
    
    return data

def get_data(dataset, use_all_samples=False, keep_blind_spot=False, make_images=False, filename=None):
    
    if dataset == 'Rotterdam':

        # Get data
        data_file = '/home/ubelix/artorg/kucur/Workspace/Data/rotterdam_data/rotterdam_numpy_data_05_08_2020.npz'
        with np.load(data_file, allow_pickle=True) as peri_data:
            vf = peri_data["vf"]
            td = peri_data["td"]
            pId = peri_data["pid"]
            nv = peri_data["nv"]
            ages = peri_data['ages']
            eyeId = peri_data['eyes']
        n_samples = len(vf)
        xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/p24d2.csv', skiprows=1, delimiter=',')
        td[np.isnan(td)] = -40
        td[td==-1000] = -40

        if keep_blind_spot is False:
            # Then delete blind spots
            vf = np.delete(vf, [25, 34], axis=1)
            nv = np.delete(nv, [25, 34], axis=1)
            td = np.delete(td, [25, 34], axis=1)
            xy = np.delete(xy, [25, 34], axis=0)

    elif dataset == 'Bern':

        data_file = "/home/ubelix/artorg/kucur/Workspace/Data/insel_data/insel_data_G_program_extracted_on_04_08_2020_no_nans.npz"
        with np.load(data_file, allow_pickle=True) as peri_data:
            vf = peri_data['ph1']
            pId = peri_data['pid']
            nv = peri_data['nv']
            ages = peri_data['ages']
            eyeId = peri_data['eyes']
            xy = peri_data['xy']
        td = vf - nv
        n_samples = len(vf)
        # md = -1 * np.load(data_file)['md']
        eyeId = (eyeId == "OS ").astype('uint8')
        # xy = np.loadtxt('/home/ubelix/artorg/kucur/Workspace/Data/Gpattern.csv', skiprows=1, delimiter=',')

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
    if use_all_samples:
        for each_id in pId_u:
            for e in [0, 1]:
                idx = np.where((pId == each_id) & (eyeId == e))[0]
                aux = ages[idx]
                if np.all(np.diff(aux) >= 0): # Should be sorted!
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
                        data.age_curr.append(ages[idx[k]])
                        data.age_next.append(ages[idx[k+1]])
                        data.labels_binary.append((groups_num[idx[k+1]] - groups_num[idx[k]]) >= 1) # 1 if prgressed, 0 otherwise
                else:
                    print("{:.0f}-{:.0f}".format(each_id, e))
    else:
        for each_id in pId_u:
            for e in [0, 1]:
                idx = np.where((pId == each_id) & (eyeId == e))[0]
                aux = ages[idx]
                if np.all(np.diff(aux) >= 0): # Should be sorted!
                    for k in range(len(idx)-1):
                        time_diff = (ages[idx[k+1]] - ages[idx[k]])/30.0
                        if ((time_diff < 13) & (time_diff > 4)):
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
                            data.age_curr.append(ages[idx[k]])
                            data.age_next.append(ages[idx[k+1]])
                            data.labels_binary.append((groups_num[idx[k+1]] - groups_num[idx[k]]) >= 1) # 1 if prgressed, 0 otherwise
                else: # not sorted
                    print("{:.0f}-{:.0f}".format(each_id, e))
    data.xy = xy
                
    # Make arrays from the lists
    data.make_arrays()

    # Make images if True
    if make_images:
        data.vor_images = generate_voronoi_images_given_image_size(data.td, xy)
        data.vor_images_next = generate_voronoi_images_given_image_size(data.td_next, xy)

    if filename is not None:
        save_object(data, filename)

    return data

def prepare_data(data, vf_format="straight", add_md=False):
    
    # Split based on the patients
    pid_indices = np.unique(data.pid)
    pid_train, pid_test = train_test_split(pid_indices, test_size=0.15, random_state=42)
    pid_train, pid_val = train_test_split(pid_train, test_size=0.15, random_state=30)
    
    indices_train = np.where(np.isin(data.pid, pid_train))[0]
    indices_val = np.where(np.isin(data.pid, pid_val))[0]
    indices_test = np.where(np.isin(data.pid, pid_test))[0]

    # In which format, we want to have our data
    if vf_format == "straight":

        x_train = data.td[indices_train]
        x_val = data.td[indices_val]
        x_test = data.td[indices_test]
        y_train = data.labels_binary[indices_train]
        y_val = data.labels_binary[indices_val]
        y_test = data.labels_binary[indices_test]

        # Add current age and time difference to the input
        time_diffs = data.age_next - data.age_curr
        x_train = np.concatenate([x_train, data.age_curr[indices_train, None], time_diffs[indices_train, None]], axis=1)
        x_val = np.concatenate([x_val, data.age_curr[indices_val, None], time_diffs[indices_val, None]], axis=1)
        x_test = np.concatenate([x_test, data.age_curr[indices_test, None], time_diffs[indices_test, None]], axis=1)

        if add_md:
            x_train = np.concatenate([x_train, data.md_curr[indices_train, None]], axis=1)
            x_val = np.concatenate([x_val, data.md_curr[indices_val, None]], axis=1)
            x_test = np.concatenate([x_test, data.md_curr[indices_test, None]], axis=1)

        # scaler = StandardScaler() # MinMaxScaler(feature_range=[-1, 1])
        # scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_val = scaler.transform(x_val)
        # x_test = scaler.transform(x_test)
    
    elif vf_format == "voronoi":

        # vor_imgs = generate_voronoi_images_given_image_size(data.td, data.xy)

        # Add current age and time difference as additional channels
        sh1 = data.vor_images.shape[1]
        sh2 = data.vor_images.shape[2]
        time_diffs = data.age_next - data.age_curr
        age_mat = np.moveaxis(np.tile(data.age_curr, [sh1, sh2, 1]), -1, 0)
        time_mat = np.moveaxis(np.tile(time_diffs, [sh1, sh2, 1]), -1, 0)
        vor_data = np.stack([data.vor_images, age_mat, time_mat], axis=1)

        # Split data
        x_train = vor_data[indices_train]
        x_val = vor_data[indices_val]
        x_test = vor_data[indices_test]
        y_train = data.labels_binary[indices_train]
        y_val = data.labels_binary[indices_val]
        y_test = data.labels_binary[indices_test]

        # Normalize channels independently
        x_train, mean_tr, std_tr = standardize_image(x_train)
        x_val, _, _ = standardize_image(x_val, mean_tr, std_tr)
        x_test, _, _ = standardize_image(x_test, mean_tr, std_tr)

    return x_train, y_train, x_val, y_val, x_test, y_test, indices_train, indices_val, indices_test

# NETWORK

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

def define_ae(d_in, d_out, d_hidden=256):

    # model = torch.nn.Sequential(
    #       torch.nn.Linear(d_in, ),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(4, 4),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(4, 4),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(4, 4),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(4, 128),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(128, 128),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(128, 64),
    #       torch.nn.ReLU(),
    #       torch.nn.Linear(64, d_out),
    #       torch.nn.LogSoftmax(dim=1)
    #     )


    model = torch.nn.Sequential(
          torch.nn.Linear(d_in, d_hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(d_hidden, d_hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(d_hidden, d_hidden),
          torch.nn.ReLU(),
          torch.nn.Linear(d_hidden, d_out),
        )

    return model

class simple_conv_net(torch.nn.Module):
    def __init__(self, d_out=1, input_size=(61, 61), batch_size=32):
        super(simple_conv_net, self).__init__()
        conv_kernel_size = 3
        pool_kernel_size = 2
        self.batch_size = batch_size
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 8, 3)
        self.conv3 = torch.nn.Conv2d(8, 4, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Compute the size
        in_size = input_size
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=conv_kernel_size)
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=pool_kernel_size, stride=pool_kernel_size)
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=conv_kernel_size)
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=pool_kernel_size, stride=pool_kernel_size)
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=conv_kernel_size)
        in_size = utils_pytorch.conv2d_output_shape(in_size, kernel_size=pool_kernel_size, stride=pool_kernel_size)

        ch_size = self.conv3.out_channels
        self.fc1 = torch.nn.Linear(ch_size * in_size[0] * in_size[1], 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, d_out)
        # self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):

        current_batch_size = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(current_batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.logsoftmax(x)

        return x


# VISUALIZE

def generate_voronoi_images_given_image_size(data, xy_coordinates, image_size=(61, 61)):

    """ 
    Generates voronoi images using given values for patching colors and given coordiantes for seed points.
    Here image size is fixed or given by the user.
    Output conventions according to Theano CNN implemntation (3D matrix -- (number_of_samples, image_row, image_col))
    
    """

    # Number of seed locations/points
    num_locs = xy_coordinates.shape[0]
    num_obs = data.shape[0]

    # start from 0
    x = np.zeros((num_locs, 1))
    y = np.zeros((num_locs, 1))
    x[:, 0] = xy_coordinates[:, 0]
    y[:, 0] = xy_coordinates[:, 1]
    x = x + int(image_size[0]/2)
    y = y + int(image_size[1]/2)
    voronoi_points = np.column_stack((x, y))

    # A grid of full space points including seed ones
    space_coordinates = np.mgrid[0:image_size[0], 0:image_size[1]]
    x_coord = space_coordinates[0, :].flatten() # columns
    y_coord = space_coordinates[1, :].flatten() # rows
    space_coordinates = np.vstack((x_coord, y_coord)).transpose()

    # Define an image
    img_col_size = image_size[0]
    img_row_size = image_size[1]
    img = np.zeros((img_row_size, img_col_size))

    # Fill in image
    vor_images = np.zeros((num_obs, img_row_size, img_col_size))
    for k in range(num_obs):
        value_vector = data[k, :]
        for img_col_ind, img_row_ind in space_coordinates:
            dist = (voronoi_points[:, 0] - img_col_ind)**2 + (voronoi_points[:, 1] - img_row_ind)**2
            idx = np.argmin(dist)
            img[img_row_ind, img_col_ind] = value_vector[idx]

        # Have to flip because of matrix conventions (y axis coordinates increase
        # from down to up but a matrix row indices increase from up to down)
        img = np.flipud(img)
        vor_images[k, : , :] = img

    return vor_images
