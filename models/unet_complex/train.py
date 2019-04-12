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
from models.unet_complex.unet_model import UnetModel
from data.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os 
SMOOTH = bool(os.environ['SMOOTH']) if 'SMOOTH' in os.environ else False
TRAIN_COMPLEX = bool(os.environ['TRAIN_COMPLEX']) if 'TRAIN_COMPLEX' in os.environ else False
RENORM = bool(os.environ['RENORM']) if 'RENORM' in os.environ else True


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
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        if self.use_mask:
            mask = transforms.get_mask(kspace, self.mask_func, seed)
            masked_kspace = mask * kspace
        else:
            masked_kspace = kspace
        
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))

        image_abs = transforms.complex_abs(image)
        _, mean_abs, std_abs = transforms.normalize_instance(image_abs, eps=1e-11)
        # Normalize input
        image, mean, std = transforms.normalize_instance_complex(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = transforms.to_tensor(target)
        target_train = target

        if not TRAIN_COMPLEX:
            # Normalize target
            if RENORM:
                target_train = transforms.normalize(target, mean_abs, std_abs, eps=1e-11)
                target_train = target_train.clamp(-6, 6) # Return target (for viz) and target_clamped (for training)
        else:
            target_train = transforms.ifft2(kspace)
            target_train = transforms.complex_center_crop(target_train, (self.resolution, self.resolution))

            if RENORM:
                target_train = transforms.normalize(target_train, mean, std, eps=1e-11)
        return image, target_train, mean, std, attrs['norm'].astype(np.float32), target, mean_abs, std_abs

def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    train_transform = DataTransform(train_mask, args.resolution, args.challenge) 
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True) 
    test_transform = DataTransform(None, args.resolution, args.challenge, use_seed=True, use_mask=False) 
    return train_transform, val_transform, test_transform

def train_step(model, data, device):
    input, target, mean, std, norm, _, mean_abs, std_abs = data
    input = input.to(device)
    target = target.to(device)
    output = model(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    mean = mean.to(device)
    std = std.to(device)

    if TRAIN_COMPLEX and not RENORM:
        output = transforms.unnormalize(output, mean, std)

    elif not TRAIN_COMPLEX:
        output = transforms.unnormalize(output, mean, std)
        output = transforms.complex_abs(output)
        if RENORM: 
            mean_abs = mean_abs.unsqueeze(1).unsqueeze(2).to(device)
            std_abs = std_abs.unsqueeze(1).unsqueeze(2).to(device)
            output = transforms.normalize(output, mean_abs, std_abs)

    loss_f = F.smooth_l1_loss if SMOOTH else F.l1_loss
    loss = loss_f(output, target)
    if RENORM:
        return loss
    else:
        return 1e9 * loss

def inference(model, data, device):
    input, _, mean, std, _, target, mean_abs, std_abs = data
    input = input.to(device)
    target = target.to(device)
    output = model(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    mean = mean.to(device)
    std = std.to(device)
    mean_abs = mean_abs.unsqueeze(1).unsqueeze(2).to(device)
    std_abs = std_abs.unsqueeze(1).unsqueeze(2).to(device)
    
    output = transforms.unnormalize(output, mean, std)
    output = transforms.complex_abs(output)
    return output, target

def build_model(args):
    model = UnetModel(
        in_chans=2,
        out_chans=2,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model

def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer
