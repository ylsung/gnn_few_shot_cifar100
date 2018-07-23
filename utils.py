import os
import logging

import numpy as np

import torch
from torch.utils.data import Dataset

def create_logger(save_path='', file_type=''):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(logging.INFO)

        logger.addHandler(fh)

    return logger

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class self_Dataset(Dataset):
    def __init__(self, data, label=None, transform=None):
        super(self_Dataset, self).__init__()

        self.data = data
        self.label = label
        self.transform = transform
    def __getitem__(self, index):
        data = self.data[index]
        # data = np.moveaxis(data, 3, 1)
        # data = data.astype(np.float32)

        data = self.transform(data)
        if self.label is not None:
            label = self.label[index]
            # print(label)
            # label = torch.from_numpy(label)
            # label = torch.LongTensor([label])
            return data, label
        else:
            return data, 1
    def __len__(self):
        return len(self.data)