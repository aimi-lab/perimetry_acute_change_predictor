
import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
from predict_acute_change import get_data

def generate_fun(inputs, targets, balanced=True):
    
    assert len(inputs) == len(targets)

    # Indices of samples
    N = len(inputs)
    indices = np.arange(N)

    import IPython
    IPython.embed()

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

data = get_data('bern')
inputs = data.td
targets = data.labels_num

inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
targets = torch.from_numpy(targets).type(torch.FloatTensor)

generate_fun(inputs, targets, balanced=True)