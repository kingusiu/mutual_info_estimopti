import torch

def make_tensor_dataset(*np_arrays):
    '''
    Convert numpy arrays to torch tensors and save to torch dataset.
    '''
    tensors = [torch.from_numpy(arr) for arr in np_arrays]
    tensors = [t.unsqueeze(1) if t.ndim == 1 else t for t in tensors]
    return torch.utils.data.TensorDataset(*tensors)

