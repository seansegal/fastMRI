import numpy as np
import matplotlib.pyplot as plt


def normalize(img):
    img -= img.min()
    return img / img.max()


class DisplaySlices():

    def __init__(self, slices):
        self.slices = slices  # List of tuples
        self.outputs = []
        self.targets = []

    def __call__(self, volume_dict, model_name=None):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        for slice in range(volume_dict['output'].shape[0]):
            if (volume_dict['volume_id'], slice) in self.slices:
                self.outputs.append(volume_dict['output'][slice, :, :])
                self.targets.append(volume_dict['target'][slice, :, :])

    def plot_figure(self):
        # Normalize
        for output, target in zip(self.outputs, self.targets):
            output = normalize(output)
            target = normalize(target)

        fig = plt.figure(figsize=(15, 5))
        outputs = np.array(self.outputs)
        targets = np.array(self.targets)
        ax = plt.subplot(2, 1, 1)
        ax.set_ylabel('Reconstructions')
        ax.set_xticks([])
        ax.set_yticks([])
        outputs = outputs.transpose((1, 0, 2)).reshape(outputs.shape[1], -1)
        plt.imshow(outputs)
        plt.ylabel('Outputs')
        ax = plt.subplot(2, 1, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        targets = targets.transpose((1, 0, 2)).reshape(targets.shape[1], -1)
        plt.imshow(targets)
        plt.ylabel('Targets')
        plt.tight_layout()
        return fig
