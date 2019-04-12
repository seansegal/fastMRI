import matplotlib.pyplot as plt


def normalize(img):
    img -= img.min()
    return img / img.max()


class VisualizeNormalModel():

    def __init__(self, index):
        self.index = index
        self.counter = 0 
        self.outputs = {}
        self.targets = {}
        self.inputs = None

    def __call__(self, volume_dict, model_name):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        for s in range(volume_dict['output'].shape[0]):
            if self.counter == self.index:
                if model_name not in self.outputs:
                    self.outputs[model_name] = []
                    self.targets[model_name] = []
                self.outputs[model_name].append(volume_dict['output'][s])
                self.targets[model_name].append(volume_dict['target'][s])
                if model_name == 'UNet':
                    self.inputs = volume_dict['input'][s]
            self.counter += 1

    def reset_counter(self):
        self.counter = 0

    def show_images(self):
        fig = plt.figure(figsize=(10, 6))
        num_models = len(self.outputs)
        for i, model in enumerate(self.outputs.keys()):
            output = normalize(self.outputs[model][0])
            target = normalize(self.targets[model][0])
            inpt = normalize(self.inputs)

            ax = plt.subplot(num_models, 3, 1 + i * 3)
            if i == 0:
                plt.title('Input')
            plt.imshow(inpt)
            plt.ylabel(model)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(num_models, 3, 2 + i * 3)
            if i == 0:
                plt.title('Target')
            plt.imshow(target)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(num_models, 3, 3 + i * 3)
            if i == 0:
                plt.title('Reconstruction')
            plt.imshow(output)
            ax.set_xticks([])
            ax.set_yticks([])

        return fig
