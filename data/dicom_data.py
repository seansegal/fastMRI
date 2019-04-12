"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random

import pydicom

from torch.utils.data import Dataset
import torch

class SliceDICOM(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (str): Path to the dataset.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        self.transform = transform
        self.examples = []

        dcm_slices = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, 'knee_mri_clinical_seq_batch2')):
            if not dirnames:
                # print(dirpath, "has 0 subdirectories and", len(filenames), "files")
                fnames = [os.path.join(dirpath, filename) for filename in filenames if ".dcm" in filename.lower()]
                if len(fnames) > 0:
                    dcm_slices.append(fnames)

        if sample_rate < 1:
            random.shuffle(dcm_slices)
            num_volumes = round(len(dcm_slices) * sample_rate)
            dcm_slices = dcm_slices[:num_volumes]

        for l in dcm_slices:
            num_slices = len(l)
            self.examples += [l[slice] for slice in range(num_slices)]
        
        # print(f'Length of DICOM data used: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        ds = pydicom.dcmread(fname)
        slice_np = ds.pixel_array.astype('float32') 
        return self.transform(slice_np)
        
def DICOM_collate(batch):
    batch = [image for image in batch if image is not None]
    if len(batch) == 0:
        return None
    return torch.stack(batch, 0)


