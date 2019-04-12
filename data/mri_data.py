"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import h5py
from torch.utils.data import Dataset
import torch


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, overfit=False, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            dims = kspace.shape[1:]
            self.examples += [(fname, slice, dims) for slice in range(num_slices)]
            if overfit:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, _ = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            result = list(self.transform(kspace, target, data.attrs, fname.name, slice))
            return result


class MultiSliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, num_volumes=3, overfit=False, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        self.num_volumes = num_volumes
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice, range(slice - num_volumes // 2, slice + num_volumes//2 + 1)) for slice in range(0, num_slices)]
            if overfit:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, middle_slice, slice_range = self.examples[i]
        with h5py.File(fname, 'r') as data:
            input_images = []
            for i in slice_range:
                idx = min(max(0, i), len(data['kspace']) - 1 )
                kspace = data['kspace'][idx]
                target = data[self.recons_key][idx] if self.recons_key in data else None
                if idx == middle_slice:
                    image, unclamped_target, mean, std, norm, target_ret = self.transform(kspace, target, data.attrs, fname.name, idx)
                else:
                    image, _, _, _, _, _ = self.transform(kspace, target, data.attrs, fname.name, idx)
                input_images.append(image.unsqueeze(2))
            images = torch.cat(input_images, dim=2)
            images = images.permute((2, 0, 1))
        return images, unclamped_target, mean, std, norm, target_ret
