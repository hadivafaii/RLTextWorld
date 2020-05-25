import numpy as np
import torch
from torch.utils.data import Dataset


def transpose_send_to_cuda(data_tuple, device, dtype=torch.long):
    return tuple(map(lambda z: z.T.to(device).type(dtype), data_tuple))


class OfflineDataset(Dataset):
    def __init__(self, data_tuple, pretrain_mode, transform=None):
        assert len(data_tuple) == 4, "4 = 3 input ids + 1 labels"

        _shapes = [item.shape for item in data_tuple]
        assert _shapes.count(_shapes[0]) == len(_shapes), "Data matrices must have the same shape"

        self.data_tuple = data_tuple
        self.pretrain_mode = pretrain_mode
        self.transform = transform

    def __len__(self):
        return len(self.data_tuple[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = tuple(map(lambda z: z[idx % len(self)], self.data_tuple))

        if self.transform:
            sample = tuple(map(lambda z: self.transform(z), sample))

        return sample
