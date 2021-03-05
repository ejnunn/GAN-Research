'''
Class definitions for SPA-GAN and all related NN blocks.
'''

import torch
from torch import nn
from tqdm.auto import tqdm # progress bars
import torchvisio
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
from GAN_Research.losses import *
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

    # attn_map = x13.clone()
    attn_map_norm = attn_map_norm_and_upsample(x13, target_shape)

    return self.tanh(xn), attn_map_norm

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

    # Copied from https://github.com/szagoruyko/attention-transfer/blob/master/visualize-attention.ipynb
    # Trying to get attention map for layer x3. Is this right?
    # x3 selected because it most likely correlates to discriminative object parts
    # attn_map = x3.pow(2).mean(1) # [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

    # Take features from 2nd to last layer and create attention map 
    # attn_map = x3.clone()
    attn_map_sum = sum_of_abs_values_over_channels(x3)
    attn_map_norm = attn_map_norm_and_upsample(attn_map_sum, target_shape)
    return xn, attn_map_norm

class SPAGAN():
  '''
  SPA-GAN model ready to train on Smile/Not Smile dataset from CelebA.

  Params:
  - weights_file: 
  - model_name: 
  - dim_A: Default model to accept RGB images
  - dim_B: B&W images will be broadcast to 3 dims
  '''
  def __init__(self, weights_file=None, model_name='smile', model_dir='drive/MyDrive/GAN Research/SPAGAN/models/smile/', \
               dim_A=3, dim_B=3, load_shape=286, target_shape=256, device='cuda', lr=0.0002):
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

