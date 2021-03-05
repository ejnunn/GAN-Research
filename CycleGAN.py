'''
Class definitions for SPA-GAN and all related NN blocks.
'''

import torch
from torch import nn
from tqdm.auto import tqdm # progress bars
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # visualizations
import pandas as pd # data processing
import shutil # file management
import torch.nn.functional as F # used in Hyperparameters cell
from skimage import color
import numpy as np
import os

from GAN_Research.utils import *
from GAN_Research.loss_functions import *
from GAN_Research.layers import *

# global variable to avoid refactoring each class
target_shape = 256


class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to 
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator: 
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)
        
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn

class CycleGAN():
  '''
  SPA-GAN model ready to train on Smile/Not Smile dataset from CelebA.

  Params:
    - weights_file: 
    - model_name: 
    - dim_A: Default model to accept RGB images
    - dim_B: B&W images will be broadcast to 3 dims
  '''
  def __init__(self, weights_file=None, model_name='smile', model_dir='drive/MyDrive/GAN Research/CycleGAN/models/smile/', dim_A=3, dim_B=3, \
               load_shape=286, target_shape=256, device='cuda', lr=0.0002):
    # Model Hyperparameters
    self.weights_file = weights_file
    self.model_name = model_name
    self.model_dir = model_dir
    self.device = device
    self.dim_A = dim_A
    self.dim_B = dim_B

    # Define model architecture
    self.gen_AB = Generator(self.dim_A, self.dim_B).to(self.device)
    self.gen_BA = Generator(self.dim_B, self.dim_A).to(self.device)
    self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    self.disc_A = Discriminator(self.dim_A).to(self.device)
    self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
    self.disc_B = Discriminator(self.dim_B).to(self.device)
    self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=lr, betas=(0.5, 0.999))
    self.weights_file = weights_file # Filepath for pretrained weights
    
    # Load model weights
    if self.weights_file != None:
      self.load_weights(weights_file)
    else:
      self.random_weights()

  def random_weights(self):
    self.gen_AB = self.gen_AB.apply(self.weights_init)
    self.gen_BA = self.gen_BA.apply(self.weights_init)
    self.disc_A = self.disc_A.apply(self.weights_init)
    self.disc_B = self.disc_B.apply(self.weights_init)
  
  def weights_init(self, m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
          torch.nn.init.normal_(m.weight, 0.0, 0.02)
      if isinstance(m, nn.BatchNorm2d):
          torch.nn.init.normal_(m.weight, 0.0, 0.02)
          torch.nn.init.constant_(m.bias, 0)

  def load_weights(self, weights_file):
    self.model_dir = os.path.dirname(weights_file) # 'drive/MyDrive/GAN Research/SPAGAN/models/smile/'

    pre_dict = torch.load(weights_file)
    self.gen_AB.load_state_dict(pre_dict['gen_AB'])
    self.gen_BA.load_state_dict(pre_dict['gen_BA'])
    self.gen_opt.load_state_dict(pre_dict['gen_opt'])
    self.disc_A.load_state_dict(pre_dict['disc_A'])
    self.disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
    self.disc_B.load_state_dict(pre_dict['disc_B'])
    self.disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])

  def train(self, dataset, adv_criterion=nn.MSELoss(), recon_criterion=nn.L1Loss(), num_epochs=25, save_epoch=1, \
            fast_save=True, display_step=500, batch_size=1, is_inception=False):
    # Initialize local variables
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cur_step = 0
    
    for epoch in range(num_epochs):
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
        for real_A, real_B in tqdm(dataloader):
            ### Reshape images and load onto GPU ###
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            ### Update discriminator A ###
            self.disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = self.gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, self.disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True) # Update gradients
            # disc_A_opt.step() # Update optimizer

            ### Update discriminator B ###
            self.disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = self.gen_AB(real_A_input)
            disc_B_loss = get_disc_loss(real_B_input, fake_B, self.disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True) # Update gradients
            # disc_B_opt.step() # Update optimizer

            ### Update generator ###
            self.gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, self.gen_AB, self.gen_BA, \
                self.disc_A, self.disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            
            gen_loss.backward() # Update gradients
            
            ### Update optimizers ###
            # Error about "in-place" operations caused crashing
            # Updating optimizers AFTER all backprops fixed it
            self.disc_A_opt.step()
            self.disc_B_opt.step()
            self.gen_opt.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A_input, real_B_input]), size=(self.dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([real_A, real_B]), size=(self.dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(self.dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if fast_save:
                    torch.save({
                        'gen_AB': self.gen_AB.state_dict(),
                        'gen_BA': self.gen_BA.state_dict(),
                        'gen_opt': self.gen_opt.state_dict(),
                        'disc_A': self.disc_A.state_dict(),
                        'disc_A_opt': self.disc_A_opt.state_dict(),
                        'disc_B': self.disc_B.state_dict(),
                        'disc_B_opt': self.disc_B_opt.state_dict()
                    }, f"{self.model_name}_{cur_step}.pth")

            ### Save to Google Drive ###
            if epoch % save_epoch == 0:
              filename =  self.model_name + '_' + str(epoch)
              save_model_to_drive(self, filename)

            cur_step += 1

