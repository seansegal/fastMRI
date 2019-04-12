from data.mri_data import SliceData, MultiSliceData
from torch.utils.data import Sampler
import math


class VolumeSampler(Sampler):
    """
        Ensures each batch comes from the same volume. Used for evaluation metrics to be computed over each volume.
    """

    def __init__(self, dataset, batches_per_volume=1):
        """
        Args:
            dataset (torch.utils.data.Dataset)     SliceData dataset
            batches_per_volume (int)               Number of batches for each volume 

        Returns:
            VolumeSampler:                         A torch.utils.data.Sampler which should be passed as batched_sampler
                                                   to the Dataloader
        """
        self.volumes = {}
        self.batches_per_volume = batches_per_volume
        for idx, example in enumerate(dataset.examples):
            fname = example[0]
            indices = self.volumes.get(fname, [])
            indices.append(idx)
            self.volumes[fname] = indices

    def __iter__(self):
        for fname, indices in self.volumes.items():
            example_length = int(math.ceil(len(indices)/self.batches_per_volume))
            for batch in range(self.batches_per_volume):
                yield indices[batch*example_length:(batch + 1)*example_length]

    def __len__(self):
        return len(self.volumes) * self.batches_per_volume
