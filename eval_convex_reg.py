#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:19:29 2021

@author: subhadip
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import odl

#torch.cuda.set_device(3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import mayo_utils, convex_models, torch_wrapper
from convex_models import n_layers, n_filters, kernel_size
from train_convex_reg import clip_fbp

#from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

##############create the network models#################
acr = convex_models.ICNN(n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers).to(device)
sfb = convex_models.SFB(n_in_channels=1, n_kernels=10, n_filters=32).to(device)
l2_net = convex_models.L2net().to(device) 
acr.eval()
sfb.eval()
l2_net.eval()

### load trained models
model_path = './trained_models/'
if clip_fbp:
    acr.load_state_dict(torch.load(model_path + "icnn_clipped_fbp.pt"))
    sfb.load_state_dict(torch.load(model_path + "sfb_clipped_fbp.pt"))
    l2_net.load_state_dict(torch.load(model_path + "l2_net_clipped_fbp.pt"))
else:
    acr.load_state_dict(torch.load(model_path + "icnn.pt"))
    sfb.load_state_dict(torch.load(model_path + "sfb.pt"))
    l2_net.load_state_dict(torch.load(model_path + "l2_net.pt"))

########## check the dimensions #############
x = torch.randn(2, 1, 512, 512).to(device)
##### run a numerical convexity test ############
cvx_flag = convex_models.test_convexity(acr, x, device=device)

############ dataloaders #######################
print('creating test dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]
eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'test'),\
                              batch_size = 1, shuffle = False)

print('number of minibatches during testing = %d'%len(eval_dataloader))
############################################
######### forward operator and FBP ######################
import simulate_projections_for_train_and_test
from simulate_projections_for_train_and_test import img_size, space_range, num_angles, det_shape, noise_std_dev, geom
space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (img_size, img_size), dtype='float32', weighting=1.0)
if(geom=='parallel_beam'):
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
else:
    geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.5*space_range, \
                                                             det_radius=5.0, num_angles=num_angles, det_shape=det_shape)
    
fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
print('operator norm = {:.4f}'.format(op_norm))

fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
adjoint_op_odl = fwd_op_odl.adjoint

fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)


####### variational optimizer for the learned convex prior ####################
sq_loss = torch.nn.MSELoss(reduction='mean') #data-fidelity loss

def acr_optimizer(x_init, x_ground_truth, y_test, n_iter, lambda_acr, lr=0.80): 
    x_cvx = x_init.clone().detach().requires_grad_(True).to(device) 
    x_optimizer = torch.optim.SGD([x_cvx], lr=lr)
    x_test_np = x_ground_truth.cpu().detach().numpy()
    data_range = np.max(x_test_np) - np.min(x_test_np)
    
    for iteration in np.arange(n_iter):
        x_optimizer.zero_grad()
        y_cvx = fwd_op(x_cvx)
        data_loss = sq_loss(y_test, y_cvx)
        
        ####### compute the regularization term ############
        prior_acr = lambda_acr*acr(x_cvx).mean()
        prior_sfb = lambda_acr*sfb(x_cvx).mean()
        prior_l2 = lambda_acr*l2_net(x_cvx).mean()
        prior =  prior_acr + prior_sfb + prior_l2
        
        variational_loss = data_loss + prior

        variational_loss.backward(retain_graph=True)
        x_optimizer.step()
        #lr_scheduler.step()

        x_np = x_cvx.cpu().detach().numpy().squeeze()
        psnr = compare_psnr(np.squeeze(x_test_np),x_np,data_range=data_range)
        ssim = compare_ssim(np.squeeze(x_test_np),x_np,data_range=data_range)
        
        if(iteration%50==0):
            recon_log = '[iter: {:d}/{:d}\t PSNR: {:.4f}, SSIM: {:.4f}, var_loss: {:.6f}, regularization: ACR {:.6f}, SFB {:.6f}, l2-term {:.6f}]'\
            .format(iteration, n_iter, psnr, ssim, variational_loss.item(), prior_acr.item(), prior_sfb.item(), prior_l2.item())
            
            print(recon_log)
            
    
    x_np = x_cvx.cpu().detach().numpy().squeeze()
    psnr = compare_psnr(np.squeeze(x_test_np),x_np,data_range=data_range)
    ssim = compare_ssim(np.squeeze(x_test_np),x_np,data_range=data_range)
    return x_np, psnr, ssim


####### reconstruction of test data ##############
log_file_name = "convex_regularizer_reconstruction_log.txt"
try:
    os.remove(log_file_name)
except OSError:
    pass

log_file = open(log_file_name, "w+")
log_file.write("################ reconstruction log for convex regularizer\n ################")
  
recon_image_path = './convex_reg_recon_image/recon/'   
phantom_path = './convex_reg_recon_image/phantom/'  
fbp_path = './convex_reg_recon_image/fbp/'   

os.makedirs(phantom_path, exist_ok=True) 
os.makedirs(fbp_path, exist_ok=True) 
os.makedirs(recon_image_path, exist_ok=True)        

psnr_fbp_avg, ssim_fbp_avg, psnr_cvx_avg, ssim_cvx_avg = 0.0, 0.0, 0.0, 0.0 #### keep track of the average
num_test_images = 0

for idx, batch in enumerate(eval_dataloader):
    phantom = batch["phantom"].to(device) #true images
    #fbp = batch["fbp"].to(device) #FBP
    #sinogram = batch["sinogram"].to(device) #sinogram
    sinogram, fbp = simulate_projections_for_train_and_test.compute_projection(phantom, num_angles=num_angles, det_shape=det_shape, \
                       space_range=space_range, geom=geom, noise_std_dev=noise_std_dev)
    
    phantom_image = phantom.cpu().detach().numpy().squeeze()
    data_range = np.max(phantom_image) - np.min(phantom_image)
    fbp_image = fbp.cpu().detach().numpy().squeeze()

    #convex reg. reconstruction and FBP
    if clip_fbp:
        fbp_image = mayo_utils.cut_image(fbp_image, vmin=0.0, vmax=1.0)
        n_iter, lambda_acr = 400, 0.05
    else:
        n_iter, lambda_acr = 350, 0.04
        
        
    x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device)
    x_np_cvx, psnr_cvx, ssim_cvx = acr_optimizer(x_init, phantom, sinogram, n_iter=n_iter, lambda_acr=lambda_acr)
    
    
    psnr_fbp = compare_psnr(phantom_image,fbp_image,data_range=data_range)
    ssim_fbp = compare_ssim(phantom_image,fbp_image,data_range=data_range)
    
    recon_log = 'test-image [{:d}/{:d}]:\t FBP: PSNR {:.4f}, SSIM {:.4f}\t convex-reg: PSNR {:.4f}, SSIM {:.4f}\n'\
        .format(idx, len(eval_dataloader), psnr_fbp, ssim_fbp, psnr_cvx, ssim_cvx)
    
    print(recon_log)
    log_file.write(recon_log)
    
    ### compute running sum for average
    psnr_fbp_avg += psnr_fbp
    ssim_fbp_avg += ssim_fbp
    psnr_cvx_avg += psnr_cvx
    ssim_cvx_avg += ssim_cvx
    num_test_images += 1
    
    ### save as numpy arrays ####
    out_filename = phantom_path + 'phantom_%d'%idx + '.npy'
    np.save(out_filename, phantom_image)
    
    out_filename = fbp_path + 'fbp_%d'%idx + '.npy'
    fbp_image = mayo_utils.cut_image(fbp_image, vmin=0.0, vmax=1.0) ### readjust the range for dispaly
    np.save(out_filename, fbp_image)
    
    out_filename = recon_image_path + 'acr_recon_%d'%idx + '.npy'
    x_np_cvx = mayo_utils.cut_image(x_np_cvx, vmin=0.0, vmax=1.0) ### readjust the range for dispaly
    np.save(out_filename, x_np_cvx)
    #break
#### close log-file #### 
recon_log = 'average performance:: FBP: PSNR {:.4f}, SSIM {:.4f}\t convex-reg: PSNR {:.4f}, SSIM {:.4f}'.\
    format(psnr_fbp_avg/num_test_images, ssim_fbp_avg/num_test_images, psnr_cvx_avg/num_test_images, ssim_cvx_avg/num_test_images)   
log_file.write(recon_log)    
log_file.write(recon_log)


