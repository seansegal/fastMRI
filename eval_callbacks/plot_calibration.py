from common.evaluate import CalibrationMetrics, CALIBRATION_METRIC_FUNCS

class PlotCalibration():

    def __init__(self):
        self.metrics = CalibrationMetrics(CALIBRATION_METRIC_FUNCS)

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
                          volume_dict['sigmas'],
                          model_name)

    def produce_plot(self):
        return self.metrics.barplot_calibration()
    
    def print_calibration(self):
        return self.metrics.print_calibration()