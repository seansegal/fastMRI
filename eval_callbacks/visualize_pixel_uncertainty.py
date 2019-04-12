import matplotlib.pyplot as plt


def normalize(img):
    img -= img.min()
    return img / img.max()


class VisualizePixelUncertainty():

    def __init__(self, num_images):
        self.first = True
        self.inputs = []
        self.targets = []
        self.sigmas = []
        self.outputs = []

    def __call__(self, volume_dict, model_name):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        if self.first: 
            for s in range(volume_dict['target'].shape[0]):
                if s in [0, 6, 12, 18]:
                    self.outputs.append(volume_dict['output'][s])
                    self.targets.append(volume_dict['target'][s])
                    self.inputs.append(volume_dict['input'][s])
                    self.sigmas.append(volume_dict['sigmas'][s])
            self.first = False

    def show_images(self):
        figures = []
        for output, target, sigma, inpt in zip(self.outputs, self.targets, self.sigmas, self.inputs):
            output = normalize(output)
            target = normalize(target)
            sigma = normalize(sigma)
            inpt = normalize(inpt)

            fig = plt.figure(figsize=(5, 5))
            plt.subplot(2, 2, 1)
            plt.axis('off')
            plt.title('Input')
            plt.imshow(inpt)

            plt.subplot(2, 2, 2)
            plt.title('Target')
            plt.axis('off')
            plt.imshow(target)

            plt.subplot(2, 2, 3)
            plt.title('Reconstruction')
            plt.axis('off')
            plt.imshow(output)

            plt.subplot(2, 2, 4)
            plt.axis('off')
            plt.title('Uncertainty (Variance)')
            plt.imshow(sigma)
            figures.append(fig)

        return figures
