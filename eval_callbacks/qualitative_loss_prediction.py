import numpy as np
import matplotlib.pyplot as plt


class QualitativeLossPrediction():

    def __init__(self):
        self.confidences = {}
        self.indexes = {}

    def __call__(self, volume_dict, model_name=None):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        model_name = model_name if model_name is not None else volume_dict['model_name']
        if model_name not in self.confidences:
            self.confidences[model_name] = []
            self.indexes[model_name] = []
        self.confidences[model_name] += volume_dict['slice_confidences'].tolist()
        self.indexes[model_name] += [(volume_dict['volume_id'], slice)
                                     for slice, _ in enumerate(volume_dict['slice_confidences'])]

    def get_least_confident(self, x):
        for model in self.confidences:
            conf_order = np.array(self.confidences[model]).argsort()
            to_return = []
            for idx in conf_order[:x]:
                to_return.append(self.indexes[model][idx])
            return to_return

    def get_most_confident(self, x):
        for model in self.confidences:
            conf_order = np.array(self.confidences[model]).argsort()
            to_return = []
            for idx in conf_order[-x:]:
                to_return.append(self.indexes[model][idx])
            return to_return

    def produce_confidences_histogram(self):
        figures = []
        for model in self.confidences:
            fig = plt.figure()
            plt.title('Confidence Scores ({})'.format(model))
            plt.hist(self.confidences[model])
            figures.append(fig)
        return figures
