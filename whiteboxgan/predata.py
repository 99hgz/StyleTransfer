import logging
from os.path import splitext
from os import listdir

import numpy as np
import torch
from torch.utils.data import Dataset

from glob import glob
from PIL import Image



class Data_set(Dataset):
    def __init__(self, dir_fake, dir_real, scale=1):
        self.dir_fake = dir_fake
        self.dir_real = dir_real
        self.scale=scale
        self.name1 = [splitext(file)[0] for file in listdir(
            dir_fake) if not file.startswith('.')]
        self.name2 = [splitext(file)[0] for file in listdir(
            dir_real) if not file.startswith('.')]
        self.len1=len(self.name1)
        self.len2=len(self.name2)

    def __len__(self):
        return max(self.len1,self.len2)
    @classmethod
    def resize(cls, img):
        a = np.array(img)
        if len(a.shape) == 2:
            a = np.expand_dims(a, axis=2)
        a = a.transpose((2, 0, 1))
        if a.max() > 1:
            a = a/255
        return a

    def __getitem__(self, i):
        file_fake = self.dir_fake + self.name1[i%self.len1] + '.jpg'
        file_real = self.dir_real + self.name2[i%self.len2] + '.jpg'
        try:
            fake = Image.open(file_fake)
            real = Image.open(file_real)
        except:
            print(file_fake)
            print(file_real)
        fake = self.resize(fake)
        real = self.resize(real)
        return {
            'fake': torch.from_numpy(fake).type(torch.FloatTensor),
            'real': torch.from_numpy(real).type(torch.FloatTensor),
        }  # return a dict
