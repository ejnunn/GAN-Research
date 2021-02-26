'''
Helper functions to train SPAGAN, visualize results, and save model to Google Drive.
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

# Compute the number of trainable parameters
def total_trainable_parameters(models):
  total = sum([len(x.parameters()) for x in models])
  return total

# Visualize images in a 2x2 grid.
# real_A | real_B
# fake_A | fake_B
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
  '''
  Function for visualizing images: Given a tensor of images, number of images, and
  size per image, plots and prints the images in an uniform grid.
  '''
  image_tensor = (image_tensor + 1) / 2
  image_shifted = image_tensor
  image_unflat = image_shifted.detach().cpu().view(-1, *size)
  image_grid = make_grid(image_unflat[:num_images], nrow=5)
  plt.imshow(image_grid.permute(1, 2, 0).squeeze())
  plt.show()
    
# Save backup model weights to Google Drive account.
def save_model_to_drive(model, drive_dir, filename):
  try:
    # Copy latest checkpoint to Drive for permenant storage
    save_filename = drive_dir + filename
    torch.save({
      'gen_AB': model.gen_AB.state_dict(),
      'gen_BA': model.gen_BA.state_dict(),
      'gen_opt': model.gen_opt.state_dict(),
      'disc_A': model.disc_A.state_dict(),
      'disc_A_opt': model.disc_A_opt.state_dict(),
      'disc_B': model.disc_B.state_dict(),
      'disc_B_opt': model.disc_B_opt.state_dict()
    }, f"{save_filename}.pth")
  except:
    print('Need to mount a Google Drive account to runtime.')

def sum_of_abs_values_over_channels(tensor):
  '''
  Returns the sum of absolute values of activation maps in each
  spatial location in a layer across the channel dimention (dim=1)
  Parameters:
      tensor: PyTorch 4D-tensor of shape [batch_size, channels, height, width]
  '''
  abs_tensor = torch.abs(tensor)
  sum_tensor = torch.sum(abs_tensor, dim=1)
  return sum_tensor.unsqueeze(0)

def attn_map_norm_and_upsample(tensor, output_size):
  '''
  Return the normalized and upsampled attention map tensor.
  '''
  max_value = torch.max(tensor).item()
  normalized = torch.div(tensor, max_value)

  upsample = nn.Upsample(size=output_size)
  return upsample(normalized)

