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
from models.unet_v2v.unet_model import UnetModel
from data.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import h5py
SMOOTH = bool(os.environ['SMOOTH']) if 'SMOOTH' in os.environ else False
CLAMP = bool(os.environ['CLAMP']) if 'CLAMP' in os.environ else True
from common.file_wrapper_transform import FileWrapperTransform

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, use_mask=True, normalize=True):
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
        self.normalize = normalize

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
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)

        # Normalize input
        if self.normalize:
            image, mean, std = transforms.normalize_instance(image, eps=1e-11)
            if CLAMP:
                image = image.clamp(-6, 6)
        else:
            mean = -1.0
            std = -1.0

        # Normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target_train = target
            if self.normalize:
                target_train = transforms.normalize(target, mean, std, eps=1e-11)
                if CLAMP:
                    target_train = target_train.clamp(-6, 6) # Return target (for viz) and target_clamped (for training)
            norm = attrs['norm'].astype(np.float32)
        else:
            target_train = []
            target = []
            norm = -1.0
        image_updated = []
        if os.path.exists(
                '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_train/' + fname):
            updated_fname = '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_train/' + fname
            with h5py.File(updated_fname, 'r') as data:
                image_updated = data['reconstruction'][slice]
                image_updated = transforms.to_tensor(image_updated)
        elif os.path.exists(
                '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_val/' + fname):
            updated_fname = '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_val/' + fname
            with h5py.File(updated_fname, 'r') as data:
                image_updated = data['reconstruction'][slice]
                image_updated = transforms.to_tensor(image_updated)
        elif os.path.exists(
                '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_test/' + fname):
            updated_fname = '/home/manivasagam/code/fastMRIPrivate/models/unet_volumes/reconstructions_test/' + fname
            with h5py.File(updated_fname, 'r') as data:
                image_updated = data['reconstruction'][slice]
                image_updated = transforms.to_tensor(image_updated)

        return image, target_train, mean, std, norm, target, image_updated

def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    train_transform = DataTransform(train_mask, args.resolution, args.challenge, use_seed=True, use_mask=True, normalize=False)
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True, use_mask=True, normalize=False)
    test_transform = FileWrapperTransform(DataTransform(None, args.resolution, args.challenge, use_seed=True, use_mask=False, normalize=False))
    return train_transform, val_transform, test_transform

def train_step(model, data, device):
    input, target, mean, std, norm, _, image_updated = data
    # input, mean, std = transforms.normalize_instance(input, eps=1e-11)
    target, _, _ = transforms.normalize_instance(target, eps=1e-11)
    if CLAMP:
        target = target.clamp(-6, 6)
    if len(image_updated) != 0:
        input = image_updated
    input, mean, std = transforms.normalize_instance(input, eps=1e-11)
    if CLAMP:
        input = input.clamp(-6, 6)
    input = input.unsqueeze(0).unsqueeze(1).to(device)
    target = target.to(device)
    output = model(input).squeeze(1).squeeze(0)
    
    if SMOOTH:
        loss = F.smooth_l1_loss(output, target)
    else:
        loss = F.l1_loss(output, target)
    return loss

def inference(model, data, device):
    input, target, mean, std, norm, unnormalized_target, image_updated = data
    if len(target) != 0:
        target, _, _ = transforms.normalize_instance(target, eps=1e-11)
    if len(image_updated) != 0:
        input = image_updated
    input, mean, std = transforms.normalize_instance(input, eps=1e-11)
    if CLAMP:
        input = input.clamp(-6, 6)

    input = input.unsqueeze(0).unsqueeze(1).to(device)
    if len(unnormalized_target) != 0:
        unnormalized_target = unnormalized_target.to(device)
    output = model(input).squeeze(1).squeeze(0)

    mean = mean.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
    std = std.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
    output = transforms.unnormalize(output, mean, std)
    # if len(target) != 0:
    #     target = transforms.unnormalize(target, mean, std)
    # if len(target) != 0:
    #     target = target * std + mean
    return output, unnormalized_target

def build_model(args):
    model = UnetModel(
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

def get_batch_sampler(dataset, batch_size, display=False):
    return VolumeSampler(dataset, batches_per_volume=1)
