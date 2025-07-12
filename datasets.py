import os
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import util.config as config
class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):

        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)
class DPIC_DataSet(Dataset):
    def __init__(self, mode='clients',multimodel='v',path=''):   # 
        super(DPIC_DataSet, self).__init__()
        if multimodel=='v':
            if mode =='clients':
                self.file_path='/data/EPIC/video/v_train_clients.pt'
            elif mode=='test':
                self.file_path='/data/EPIC/video/v_test.pt'
            elif mode=='common':
                self.file_path='/data/EPIC/video/v_common.pt' 
            
        elif multimodel=='a':
            if mode =='clients':
                self.file_path='/data/EPIC/audio/a_train_clients.pt'
            elif mode=='test':
                self.file_path='/data/EPIC/audio/a_test.pt'
            elif mode=='common':
                self.file_path='/data/EPIC/audio/a_common.pt' 
            
       

        self.data=torch.load(self.file_path,map_location='cpu')

        self.leng=len(self.data)
        
    def __len__(self):
        return self.leng

    def __getitem__(self, idx):
        sample=self.data[idx]   #[feature,label]
        return sample


