"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time
import datetime

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from common.args import Args
from common.tensorboard import write_metrics_to_tb
from common.evaluate import Metrics, METRIC_FUNCS
from common.subsample import MaskFunc
from data import transforms
from data.mri_data import SliceData
# from models.gan_image_fcn_consistent.gan_model import GanGenerator
from models.unet_image_fcn_consistent.unet_model import UnetModel as GanGenerator
from models.gan_image_fcn_consistent.gan_model import GanDiscriminator
from data.volume_sampler import VolumeSampler
from data.same_size_sampler import SameSizeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RENORM = True
CLAMP = False
PROJECT = True
WGAN = False


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, use_mask=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.use_mask = use_mask

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        kspace = transforms.to_tensor(kspace)
        target = transforms.ifft2(kspace)
        
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        if self.use_mask:
            mask = transforms.get_mask(kspace, self.mask_func, seed)
            masked_kspace = mask * kspace
        else:
            masked_kspace = kspace
        image = transforms.ifft2(masked_kspace)

        image_abs = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        image_abs = transforms.complex_abs(image_abs)
        image_abs, mean_abs, std_abs = transforms.normalize_instance(image_abs, eps=1e-11)

        image, mean, std = transforms.normalize_instance_complex(image, eps=1e-11)

        target = transforms.complex_center_crop(target, (320, 320))
        target = transforms.complex_abs(target)
        target_train = target
        
        if RENORM:
            target_train = transforms.normalize(target_train, mean_abs, std_abs)
            
        if CLAMP:
            image = image.clamp(-6, 6)
            target_train = target_train.clamp(-6, 6)


        return image, target_train, mean, std, mask, mean_abs, std_abs, target


def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    train_transform = DataTransform(train_mask, args.resolution, args.challenge) 
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True) 
    test_transform = DataTransform(None, args.resolution, args.challenge, use_seed=True, use_mask=False) 
    return train_transform, val_transform, test_transform


def project_to_consistent_subspace(residual, input, mask):
    reconstructed_kspace = transforms.fft2(residual)
    original_kspace = transforms.fft2(input)
    new_kspace = (1 - mask) * reconstructed_kspace + mask * original_kspace
    return transforms.ifft2(new_kspace)


def generate(generator, data, device):

    input, _, mean, std, mask, _, _, _ = data
    input = input.to(device)
    mask = mask.to(device)
    
    # Use network to predict residual
    residual = generator(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # Projection to consistent K-space
    if PROJECT:
        output = project_to_consistent_subspace(residual, input, mask)

    # Take loss on the cropped, real valued image (abs) 
    mean = mean.to(device)
    std = std.to(device)
    output = transforms.unnormalize(output, mean, std)
    output = transforms.complex_center_crop(output, (320, 320))
    output = transforms.complex_abs(output)

    return output


def train_step_generator(generator, discriminator, data, device):

    _, target, _, _, _, mean_abs, std_abs, _ = data
    target = target.to(device)
    
    output = generate(generator, data, device)
    if RENORM:
        mean_abs = mean_abs.unsqueeze(1).unsqueeze(2).to(device)
        std_abs = std_abs.unsqueeze(1).unsqueeze(2).to(device)
        output = transforms.normalize(output, mean_abs, std_abs)
    consistency_loss = F.l1_loss(output, target)

    p_output = discriminator(output.unsqueeze(1))
    if WGAN:
        disc_loss = - torch.mean(p_output)
    else:
        disc_loss = F.binary_cross_entropy_with_logits(p_output, torch.ones(p_output.shape).to(device))

    return 0.01 * disc_loss, consistency_loss


def train_step_discriminator(generator, discriminator, data, device):
    
    _, target, _, _, _, mean_abs, std_abs, _ = data
    target = target.to(device)

    output = generate(generator, data, device).detach()
    if RENORM:
        mean_abs = mean_abs.unsqueeze(1).unsqueeze(2).to(device)
        std_abs = std_abs.unsqueeze(1).unsqueeze(2).to(device)
        output = transforms.normalize(output, mean_abs, std_abs)

    # Real loss
    p_target = discriminator(target.unsqueeze(1).to(device))
    if WGAN:
        real_loss = - torch.mean(p_target)
    else:
        real_loss = F.binary_cross_entropy_with_logits(p_target, torch.ones(p_target.shape).to(device))
    
    # Fake loss
    p_output = discriminator(output.unsqueeze(1))
    if WGAN:
        fake_loss = torch.mean(p_output)
    else:
        fake_loss = F.binary_cross_entropy_with_logits(p_output, torch.zeros(p_output.shape).to(device))


    return 0.01 * real_loss, 0.01 * fake_loss, F.sigmoid(p_target), F.sigmoid(p_output)


def inference(generator, data, device):
    with torch.no_grad():

        output = generate(generator, data, device).detach()
        _, _, _, _, _, _, _, target = data
        target = target.to(device)

        return output, target

def build_model(args):
    generator = GanGenerator(
        in_chans=2,
        out_chans=2,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    discriminator = GanDiscriminator(
        in_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools+1,
        drop_prob=args.drop_prob
    ).to(args.device)
    return generator, discriminator

def build_optim(args, paramsG, paramsD):
    optimizerG = torch.optim.RMSprop(paramsG, args.lr, weight_decay=args.weight_decay)
    optimizerD = torch.optim.RMSprop(paramsD, args.lr, weight_decay=args.weight_decay)
    return optimizerG, optimizerD

def get_batch_sampler(dataset, batch_size, display=False):
    return SameSizeSampler(dataset, batch_size, display)
