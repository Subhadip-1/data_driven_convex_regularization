#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:42:13 2021

@author: subhadip
"""

import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import glob
import random
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
#a custom dataset class
class mayo_dataset(Dataset):
    def __init__(self, root, transforms_= None, aligned = True, mode = 'train'):
        self.transform = transforms.Compose(transforms_)
        self.aligned = aligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/Sinogram'% mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/FBP'% mode) + '/*.*'))
        
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/Phantom'% mode) + '/*.*'))



    def __getitem__(self, index):
        sinogram = self.transform(Image.fromarray(np.load(self.files_A[index % len(self.files_A)])))
        fbp = self.transform(Image.fromarray(np.load(self.files_C[index % len(self.files_C)])))
        
        if self.aligned:
            phantom = self.transform(Image.fromarray(np.load(self.files_B[index % len(self.files_B)])))
        else:
            phantom = self.transform(Image.fromarray(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])))
        
        

        return {'fbp': fbp, 'phantom': phantom, 'sinogram': sinogram}

    def __len__(self):
        return max([len(self.files_A), len(self.files_B), len(self.files_C)])
    
##### hard clip image to a specific interval: takes numpy array as input
def cut_image(image, vmin, vmax):
    image = np.maximum(image, vmin)
    image = np.minimum(image, vmax)
    return image