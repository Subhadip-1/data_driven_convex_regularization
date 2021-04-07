#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:39:02 2021

@author: subhadip
"""

import numpy as np
import torch
import odl
import os
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
import torch_wrapper, mayo_utils
from skimage.measure import compare_psnr


device = 'cuda' if torch.cuda.is_available() else 'cpu'

##############specify geometry parameters#################
img_size, space_range = 512, 128 #space discretization
num_angles, det_shape = 200, 400 #projection parameters
noise_std_dev = 2.0
geom = 'parallel_beam' # 'cone_beam' or 'parallel_beam': The network and optimizer hyper-parameters are optimized for 'parallel_beam'

######computing the projection#############
def compute_projection(phantom, num_angles=num_angles, det_shape=det_shape, space_range=space_range, geom=geom, noise_std_dev=noise_std_dev):
    space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (phantom.size(2), phantom.size(3)), dtype='float32', weighting=1.0)
    if(geom=='parallel_beam'):
        geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
    else:
        geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.6*space_range, \
                                                                 det_radius=1.6*space_range, num_angles=num_angles, det_shape=det_shape)
        
    
    fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
    
    fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
    fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
    
    #print('op_norm = %.4f'%(1.1 * odl.power_method_opnorm(fwd_op_odl)))
    
    sinogram = fwd_op(phantom)
    sinogram_noisy = sinogram + noise_std_dev*torch.randn(sinogram.size()).to(device)
    
    fbp = fbp_op(sinogram_noisy)
    
    return sinogram_noisy, fbp

######computing the projection#############
def compute_adjoint(sinogram, img_size=img_size, num_angles=num_angles, det_shape=det_shape, space_range=space_range, geom=geom, noise_std_dev=noise_std_dev):
    space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range], (img_size, img_size), dtype='float32', weighting=1.0)
    if(geom=='parallel_beam'):
        geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
    else:
        geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=20.0, \
                                                                 det_radius=20.0, num_angles=num_angles, det_shape=det_shape)

    fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
    adjoint_op_odl = fwd_op_odl.adjoint
    
    adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)
        
    x_adj = adjoint_op(sinogram)
    return x_adj

####################arrange the slices into training and test data ###############################
if __name__ == '__main__':
    print('creating training and test data...')
    datapath = './mayo_data/'
    output_datapath = './mayo_data_arranged_patientwise/'
    shutil.rmtree(output_datapath, ignore_errors=True) #delete pre-existing folders
    
    files = sorted(os.listdir(datapath))
    for idx in range(len(files)):
        filename = datapath + files[idx]
        image = np.load(filename)
        image = (image - image.min())/(image.max() - image.min()) #normalize range to [0,1]
        
        ###compute projection and FBP #############
        phantom = torch.from_numpy(image).view(1, 1, img_size, img_size).to(device)
        sinogram, fbp = compute_projection(phantom)
        
        
        ######save the images as numpy files###############
        sinogram_image = sinogram.cpu().numpy().squeeze()
        fbp_image = fbp.cpu().numpy().squeeze()
        
        psnr = compare_psnr(image, fbp_image, data_range=1.0)
        nmse = torch.mean((phantom - fbp)**2)/torch.mean(phantom**2)
        print('FBP: NSME = {:.6f}\t PSNR = {:.6f}'.format(nmse, psnr))
        
        #####use patient L109 for testing, rest for training
        if('L109' not in filename): 
            #####save phantom#####
            path = output_datapath + 'train/' + 'Phantom/'
            out_filename = path + 'phantom_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, image)
            
            #####save FBP#####
            path = output_datapath + 'train/' + 'FBP/'
            out_filename = path + 'fbp_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, fbp_image)
            
            #####save sinogram#####
            path = output_datapath + 'train/' + 'Sinogram/'
            out_filename = path + 'sinogram_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, sinogram_image)
        else:
            #####save phantom#####
            path = output_datapath + 'test/' + 'Phantom/'
            out_filename = path + 'phantom_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, image)
            
            #####save FBP#####
            path = output_datapath + 'test/' + 'FBP/'
            out_filename = path + 'fbp_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, fbp_image)
            
            #####save sinogram#####
            path = output_datapath + 'test/' + 'Sinogram/'
            out_filename = path + 'sinogram_%d'%idx + '.npy'
            os.makedirs(path, exist_ok=True)
            np.save(out_filename, sinogram_image)
        
        
    ####### verify training and testing dataloader
    print('creating dataloaders...')
    transform_to_tensor = [transforms.ToTensor()]
    train_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, mode = 'train'),\
                                  batch_size = 1, shuffle = True)
    
    #testing dataloader
    eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, mode = 'test'),\
                                  batch_size = 1, shuffle = True)
 

    
    
    
    
    
    
    
    
    
    


