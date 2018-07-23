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
