#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:21:08 2021

@author: subhadip
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import itertools
import os

#torch.cuda.set_device(3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = %s'%device)

import mayo_utils, convex_models
from convex_models import n_layers, n_filters, kernel_size

#from skimage.measure import compare_ssim, compare_psnr

##############create the network models#################
acr = convex_models.ICNN(n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers).to(device)
acr.initialize_weights() #initializa weights for the non-negative layers
num_params_acr = sum(p.numel() for p in acr.parameters())
x = torch.randn(1, 1, 512, 512).to(device)
#cvx_flag = convex_models.test_convexity(acr, x, device=device)
#print(cvx_flag)

sfb = convex_models.SFB(n_in_channels=1, n_kernels=10, n_filters=32).to(device)
num_params_sfb = sum(p.numel() for p in sfb.parameters())

l2_net = convex_models.L2net().to(device) 
num_params_l2 = sum(p.numel() for p in l2_net.parameters())

print('# params: ACR: {:d}, SFB: {:d}, l2-net: {:d}'.format(num_params_acr, num_params_sfb, num_params_l2))
      
############ dataloaders #######################
print('creating dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]
train_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'train'),\
                              batch_size = 1, shuffle = True)

print('number of minibatches during training = %d'%len(train_dataloader))
############################################
####### gradient penalty loss #######
import torch.autograd as autograd
def compute_gradient_penalty(network, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #validity = net(interpolates)
    validity = network(interpolates)
    fake = torch.cuda.FloatTensor(np.ones(validity.shape)).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=validity, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
########### set-up optimizers ###########
optimizer_acr = torch.optim.Adam(acr.parameters(), lr=2*1e-5, betas=(0.5, 0.99))
optimizer_sfb = torch.optim.Adam(itertools.chain(sfb.parameters(), l2_net.parameters()), lr=2*1e-5, betas=(0.5, 0.99), weight_decay=1.0)

lambda_gp = 5.0 
n_epochs, num_minibatches = 20, 20
clip_fbp = False 

########################### RUN THE FOLLOWING WHEN EXECUTED DIRECTLY ###########################

if __name__ == '__main__':
    print('======== training log for convex regularizer ==========')
    
    acr.train()
    sfb.train()
    l2_net.train()
    
    if clip_fbp:
        log_file_name = "convex_regularizer_training_log_clipped_fbp.txt"
    else:
        log_file_name = "convex_regularizer_training_log.txt"
    
    try:
        os.remove(log_file_name)
    except OSError:
        pass
    
    log_file = open(log_file_name, "w+")
    log_file.write("################ training log for convex regularizer\n ################")
    
    ################## main training loop ##################
    for epoch in range(n_epochs):
        total_loss, total_gp_loss, total_diff = 0.0, 0.0, 0.0
        for idx, batch in enumerate(train_dataloader):
            ############################################
            phantom = batch["phantom"].to(device) #true images
            fbp = batch["fbp"].to(device) #FBP
            
            if clip_fbp:
                fbp_image = mayo_utils.cut_image(fbp.cpu().detach().numpy().squeeze(), vmin=0.0, vmax=1.0)
                fbp = torch.from_numpy(fbp_image).view(fbp.size()).to(device)
            
            #### training loss ####
            diff_loss = (acr(phantom).mean() + sfb(phantom).mean() + l2_net(phantom).mean())\
            - (acr(fbp).mean() + sfb(fbp).mean() + l2_net(fbp).mean())
            
            gp_loss = compute_gradient_penalty(acr, phantom.data, fbp.data)
            loss = diff_loss + lambda_gp * gp_loss 
            
            ####### parameter update #######
            optimizer_acr.zero_grad()
            optimizer_sfb.zero_grad()
            loss.backward()
            optimizer_acr.step()
            optimizer_sfb.step()
            
            total_loss += loss.item()
            total_gp_loss += gp_loss.item()
            total_diff += diff_loss.item()
            
            ###### perform zero-clipping to preserve convexity ######
            acr.zero_clip_weights()
    
            
            if(idx % num_minibatches == num_minibatches-1):
                ####### compute avg. loss over minibatches #######
                avg_loss = total_loss/num_minibatches
                avg_gp_loss = total_gp_loss/num_minibatches
                avg_diff = total_diff/num_minibatches
                            
                #reset the losses
                total_loss, total_gp_loss, total_diff = 0.0, 0.0, 0.0
                
                
                ######### save and print log ###########
                train_log = "epoch: [{}/{}], batch: [{}/{}], avg_loss: {:.8f}, avg_gradient_penalty: {:.8f}, avg_diff: {:.8f}".\
                      format(epoch+1, n_epochs, idx+1, len(train_dataloader), avg_loss, avg_gp_loss, avg_diff)
                print(train_log)
                log_file.write(train_log)
                
                ############ keep track of individual terms in the regularizer ############
                #### response to true images/phantoms ####
                l2_term = l2_net(phantom)
                sfb_output = sfb(phantom)
                acr_output = acr(phantom)
                
                train_log = 'response to phantom: ICNN: {:.6f}, SFB: {:.6f}, l2-term: {:.6f}'.\
                    format(torch.mean(acr_output), torch.mean(sfb_output), torch.mean(l2_term))
                print(train_log)
                log_file.write(train_log)
                
                #### response to noisy images/FBP ####
                l2_term = l2_net(fbp)
                sfb_output = sfb(fbp)
                acr_output = acr(fbp)
                
                train_log = 'response to FBP: ICNN: {:.6f}, SFB: {:.6f}, l2-term: {:.6f}\n'.\
                    format(torch.mean(acr_output), torch.mean(sfb_output), torch.mean(l2_term))
                print(train_log)
                log_file.write(train_log)
        
    ############# save the models #################
    log_file.close()      
    model_path = './trained_models/'
    os.makedirs(model_path, exist_ok=True)
    if clip_fbp:
        torch.save(acr.state_dict(),  model_path + "icnn_clipped_fbp.pt") 
        torch.save(sfb.state_dict(),  model_path + "sfb_clipped_fbp.pt") 
        torch.save(l2_net.state_dict(),  model_path + "l2_net_clipped_fbp.pt") 
    else:
        torch.save(acr.state_dict(),  model_path + "icnn.pt") 
        torch.save(sfb.state_dict(),  model_path + "sfb.pt") 
        torch.save(l2_net.state_dict(),  model_path + "l2_net.pt") 
