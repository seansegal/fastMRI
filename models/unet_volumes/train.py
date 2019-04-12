import logging

import numpy as np
import torch
from torch.nn import functional as F

from common.subsample import MaskFunc
from data import transforms
from models.shared.unet_model import UnetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, use_mask=True, normalize=False):
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
        self.use_mask = use_mask
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
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
        mean = -1.0
        std = -1.0
        if self.normalize:
            image, mean, std = transforms.normalize_instance(image, eps=1e-11)
            image = image.clamp(-6, 6)

        if target is not None:
            target = transforms.to_tensor(target)
            # Normalize target
            clamped_target = target
            if self.normalize:
                clamped_target = transforms.normalize(target, mean, std, eps=1e-11)
                clamped_target = clamped_target.clamp(-6, 6)  # Return target (for viz) and target_clamped (for training)
            norm = attrs['norm'].astype(np.float32)
        else:
            clamped_target = []
            target = []
            norm = -1.0
        return image, clamped_target, mean, std, norm, target

def get_transforms(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    test_mask = None
    train_transform = DataTransform(train_mask, args.resolution, args.challenge, use_seed=True, use_mask=True, normalize=True)
    val_transform = DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True, use_mask=True, normalize=True)
    test_transform = DataTransform(None, args.resolution, args.challenge, use_seed=True, use_mask=False, normalize=True)
    return train_transform, val_transform, test_transform

def train_step(model, data, device):
    inputs, target, mean, std, norm, _ = data
    inputs, mean, std = transforms.normalize_instance(inputs, eps=1e-11)
    target, _, _ = transforms.normalize_instance(target, eps=1e-11)
    inputs = inputs.to(device)
    target = target.to(device)

    output = model(inputs).squeeze(1)
    loss = F.l1_loss(output, target)
    return loss

def inference(model, data, device):
    normalize_instance = False
    inputs, _, mean, std, norm, target = data
    if normalize_instance:
        inputs, mean, std = transforms.normalize_instance(inputs, eps=1e-11)
    inputs = inputs.to(device)

    if len(target) != 0:
        target = target.to(device)

    output = model(inputs).squeeze(1)

    mean = mean.unsqueeze(1).unsqueeze(2).to(device)
    std = std.unsqueeze(1).unsqueeze(2).to(device)
    # if len(target) != 0:
    #     target = target * std + mean
    output = output * std + mean
    return output, target

def build_model(args):
    model = UnetModel(
        in_chans=args.num_volumes,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

