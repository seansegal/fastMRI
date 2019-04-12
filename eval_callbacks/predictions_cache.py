from common.evaluate import SliceUncertaintyMetrics, METRIC_FUNCS

class PredictionsCacheCallback():

    def __init__(self):
        self.volume_dicts = []

    def __call__(self, volume_dict, model_name=None):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        self.volume_dicts.append((volume_dict, model_name))

    def run_callback(self, callback):
        for tup in self.volume_dicts:
            volume_dict, model_name = tup
            callback(volume_dict, model_name)
