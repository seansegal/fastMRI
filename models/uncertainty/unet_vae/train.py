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
from torch.utils.data import DataLoader

from common.args import Args
from common.tensorboard import write_metrics_to_tb
from common.evaluate import Metrics, METRIC_FUNCS
from common.subsample import MaskFunc
from data import transforms
from data.mri_data import SliceData
from models.unet_vae.unet_vae_model import UnetVaeModel
from data.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
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
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = transforms.get_mask(kspace, self.mask_func, seed)
        masked_kspace = mask * kspace
        
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)

        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        target = transforms.to_tensor(target)
        
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target_clamped = target.clamp(-6, 6) # Return target (for viz) and target_clamped (for training)
        return image, target_clamped, mean, std, attrs['norm'].astype(np.float32), target

def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    train_transform = DataTransform(train_mask, args.resolution, args.challenge) 
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True) 
    return train_transform, val_transform

def train_step(model, data, device):
    input, target, mean, std, norm, _ = data
    input = input.unsqueeze(1).to(device)
    target = target.to(device)
    output = model(input).squeeze(1)
    loss = F.l1_loss(output, target)
    return loss

def inference(model, data, device):
    input, _, mean, std, _, target = data
    input = input.unsqueeze(1).to(device)
    target = target.to(device)
    output = model(input).squeeze(1)

    mean = mean.unsqueeze(1).unsqueeze(2).to(device)
    std = std.unsqueeze(1).unsqueeze(2).to(device)
    
    target = transforms.unnormalize(target, mean, std)
    output = transforms.unnormalize(output, mean, std)
    return output, target

def build_model(args):
    model = UnetVaeModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model

def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer
