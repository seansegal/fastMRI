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
from models.unet_image_fcn.unet_model import UnetModel
from data.volume_sampler import VolumeSampler
from data.same_size_sampler import SameSizeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
RENORM = bool(os.environ['RENORM']) if 'RENORM' in os.environ else False
CLAMP = bool(os.environ['CLAMP']) if 'CLAMP' in os.environ else False
SMOOTH = bool(os.environ['SMOOTH']) if 'SMOOTH' in os.environ else False

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


def generate(generator, data, device):

    input, _, mean, std, mask, _, _, _ = data
    input = input.to(device)
    mask = mask.to(device)
    
    output_network = generator(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # Take loss on the cropped, real valued image (abs) 
    mean = mean.to(device)
    std = std.to(device)
    output_network = transforms.unnormalize(output_network, mean, std)
    output_network = transforms.complex_center_crop(output_network, (320, 320))
    output_network = transforms.complex_abs(output_network)

    return output_network


def train_step(model, data, device):
    _, target, _, _, _, mean_abs, std_abs, _ = data
    target = target.to(device)

    output = generate(model, data, device)
    if RENORM:
        mean_abs = mean_abs.unsqueeze(1).unsqueeze(2).to(device)
        std_abs = std_abs.unsqueeze(1).unsqueeze(2).to(device)
        output = transforms.normalize(output, mean_abs, std_abs)

    if SMOOTH:
        loss_f = F.smooth_l1_loss
    else:
        loss_f = F.l1_loss
    
    return loss_f(output, target)


def inference(model, data, device):
    with torch.no_grad():
        output = generate(model, data, device)
        _, _, _, _, _, _, _, target = data
        target = target.to(device)

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


def get_batch_sampler(dataset, batch_size, display=False):
    return SameSizeSampler(dataset, batch_size, display)
