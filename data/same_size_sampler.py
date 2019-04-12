from data.mri_data import SliceData, MultiSliceData
from torch.utils.data import Sampler
import numpy as np
import math


class SameSizeSampler(Sampler):
    """
        Ensures each batch comes from the same volume. Used for evaluation metrics to be computed over each volume.
    """

    def __init__(self, dataset, batch_size, display=False):
        """
        Args:
            dataset (torch.utils.data.Dataset)     SliceData dataset
            batches_per_volume (int)               Number of batches for each volume 

        Returns:
            VolumeSampler:                         A torch.utils.data.Sampler which should be passed as batched_sampler
                                                   to the Dataloader
        """
        self.display = display
        hist = {}
        for idx, example in enumerate(dataset.examples):
            _, _, dims = example
            dims = repr(dims)
            exs = hist.get(dims, [])
            exs.append(idx)
            hist[dims] = exs
        
        if not display:
            self.hist = hist
        else:
            k = list(hist.keys())[0]
            self.hist = {k: [hist[k][i] for i in range(0, len(hist[k]), len(hist[k]) // batch_size)]}

        self.curr_hist = {}
        for k, v in self.hist.items():
            self.curr_hist[k] = list(v)
        self.batch_size = batch_size
        self.length = 0
        for _, values in self.hist.items():
            self.length += int(math.ceil(len(values)/self.batch_size))
    

    def __iter__(self):
        while True:
            p = np.array([len(values) for _, values in self.curr_hist.items()])
            if np.sum(p) == 0:
                self.curr_hist = {}
                for k, v in self.hist.items():
                    self.curr_hist[k] = list(v)
                break
            p = p/np.sum(p)
            key= np.random.choice(list(self.curr_hist.keys()), p=p)
            if len(self.curr_hist[key]) < self.batch_size:
                to_return = list(self.curr_hist[key])
            elif self.display:
                to_return = list(self.curr_hist[key])[:self.batch_size]
            else:
                to_return = np.random.choice(self.curr_hist[key], self.batch_size, replace=False)

            for idx in to_return:
                self.curr_hist[key].remove(idx)
            yield to_return
            if self.display:
                self.curr_hist = {}
                for k, v in self.hist.items():
                    self.curr_hist[k] = list(v)
                break

    def __len__(self):
        return self.length
