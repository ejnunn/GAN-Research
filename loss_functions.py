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

def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''

    fake_X.detach()
    fake_pred, _ = disc_X(fake_X)
    fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
    real_pred, _ = disc_X(real_X)
    real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
    disc_loss = (fake_loss + real_loss) / 2.0

    return disc_loss

def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize)
    '''

    fake_Y, _ = gen_XY(real_X)
    pred_Y, _ = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(pred_Y, torch.ones_like(pred_Y))

    return adversarial_loss, fake_Y

def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''

    identity_X, _ = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)

    return identity_loss, identity_X

def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''

    cycle_X, _ = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)

    return cycle_loss, cycle_X

# Feature Map loss function for SPA-GAN
# See page 4 of SPA-GAN paper for details
# URL: https://arxiv.org/pdf/1908.06616v3.pdf

def get_fm_loss(real_X, fake_Y, gen_XY):
  '''
  Return feature map loss between real and fake feature maps.
  Parameters:
      real_fm_X: the real feature map from pile X
                 shape = [batch_size, channels, height, width]
      fake_fm_X: the generated feature map from class X
                 shape = [batch_size, channels, height, width]
      gen_XY: the generator for class X to Y; takes input images and returns the images
          transformed to class Y
  '''
  # Feature maps of real image x_a and generated image y'_a.
  real_X.detach()
  
  _, real_fm_X = gen_XY(real_X)
  _, fake_fm_Y = gen_XY(fake_Y)

  total = 0
  for i in range(real_fm_X.shape[1]):
    # i-th feature maps
    fm_X_i = real_fm_X[0][i] # [0] <- batch size of one
    fm_Y_i = fake_fm_Y[0][i]

    # L1 norm of the difference.
    total = total + torch.linalg.norm(fm_X_i - fm_Y_i)

  # Finally, average over number of feature maps (C) in the given layer of G.
  fm_loss = torch.div(total, real_fm_X.shape[1])

  return fm_loss

def get_gen_loss(real_A, real_B, real_A_input, real_B_input, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10, lambda_fm=0.1):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
        lambda_fm: the weight of the feature-map loss
    '''

    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss,fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adv_loss_2,fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    
    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    ide_loss,identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    ide_loss_2,identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cyc_loss,cycle_A = get_cycle_consistency_loss(real_A_input, fake_B, gen_BA, cycle_criterion)
    cyc_loss_2,cycle_B = get_cycle_consistency_loss(real_B_input, fake_A, gen_AB, cycle_criterion)
    
    # Feature-Map Loss -- get_fm_loss(real_X, fake_Y, gen_XY)
    fm_loss = get_fm_loss(real_A, fake_B, gen_AB)
    fm_loss_2 = get_fm_loss(real_B, fake_A, gen_BA)
    
    # Total loss
    gen_loss = adv_loss + adv_loss_2 + lambda_identity * (ide_loss + ide_loss_2) +  \
          lambda_cycle * (cyc_loss + cyc_loss_2) + lambda_fm * (fm_loss + fm_loss_2)

    return gen_loss, fake_A, fake_B
