from common.evaluate import ArrayMetrics, ARRAY_METRIC_FUNCS

class PlotMetricNorm():

    def __init__(self):
        self.metrics = ArrayMetrics(ARRAY_METRIC_FUNCS)

    def __call__(self, volume_dict, model_name):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """

        self.metrics.push(volume_dict['target'], 
                          volume_dict['output'],
                          model_name)

    def produce_plot(self):
        return self.metrics.scatter_metrics_norm()