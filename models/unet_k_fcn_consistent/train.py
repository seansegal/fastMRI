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
from models.unet_k_fcn_consistent.unet_model import UnetModel
from data.volume_sampler import VolumeSampler
from data.same_size_sampler import SameSizeSampler
from torch.utils.data.dataloader import default_collate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        image = transforms.ifft2(masked_kspace)
        _, mean_image, std_image = transforms.normalize_instance(image, eps=1e-11)
        masked_kspace, mean, std = transforms.normalize_instance_complex(masked_kspace)
        kspace = transforms.normalize(kspace, mean, std)
        
        return masked_kspace, kspace, mean, std, mean_image, std_image, mask


def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    train_transform = DataTransform(train_mask, args.resolution, args.challenge) 
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True) 
    test_transform = DataTransform(None, args.resolution, args.challenge, use_seed=True, use_mask=False) 
    return train_transform, val_transform, test_transform


def train_step(model, data, device):
    input, target, mean, std, mean_image, std_image, mask = data
    input = input.to(device)
    mask = mask.to(device)
    target = target.to(device)
    output = model(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # Projection to consistent K-space
    output = input * mask + (1-mask) * output
    
    # Consistent K-space loss (with the normalized output and target)
    loss_k_consistent = F.l1_loss(output, target) 

    mean = mean.to(device)
    std = std.to(device)

    target = transforms.unnormalize(target, mean, std)
    output = transforms.unnormalize(output, mean, std)

    output_image = transforms.ifft2(output)
    target_image = transforms.ifft2(target)

    output_image = transforms.complex_center_crop(output_image, (320, 320))
    output_image = transforms.complex_abs(output_image)
    target_image = transforms.complex_center_crop(target_image, (320, 320))
    target_image = transforms.complex_abs(target_image)
    mean_image = mean_image.unsqueeze(1).unsqueeze(2).to(device)
    std_image = std_image.unsqueeze(1).unsqueeze(2).to(device)
    output_image = transforms.normalize(output_image, mean_image, std_image)
    target_image = transforms.normalize(target_image, mean_image, std_image)
    target_image = target_image.clamp(-6, 6)
    # Consistent image loss (with the unnormalized output and target)
    loss_image = F.l1_loss(output_image, target_image)
    loss = loss_k_consistent + loss_image
    return loss

def inference(model, data, device):
    with torch.no_grad():
        input, target, mean, std, _, _, mask = data
        input = input.to(device)
        mask = mask.to(device)
        output = model(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        output = input * mask + (1-mask) * output
        target = target.to(device)

        mean = mean.to(device)
        std = std.to(device)

        output = transforms.unnormalize(output, mean, std)
        target = transforms.unnormalize(target, mean, std)

        output = transforms.ifft2(output)
        target = transforms.ifft2(target)

        output = transforms.complex_center_crop(output, (320, 320))
        output = transforms.complex_abs(output)
        target = transforms.complex_center_crop(target, (320, 320))
        target = transforms.complex_abs(target)

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
