from common.evaluate import ArrayMetrics, ARRAY_METRIC_FUNCS
import pandas as pd
import numpy as np
from decimal import Decimal

class ComputeMetricsTable():

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

    def produce_table(self):
        table_dict = dict()
        experiment_names = list(self.metrics.metrics.keys())
        for metric in ['MSE', 'NMSE', 'PSNR', 'SSIM']:
            table_dict['Experiment Name'] = experiment_names
            metric_means = [np.asarray(self.metrics.metrics[experiment_name][metric]).mean() for experiment_name in experiment_names]
            metric_stds = [np.asarray(self.metrics.metrics[experiment_name][metric]).std() for experiment_name in experiment_names]

            metric_str = ['±'.join(['{:.3E}'.format(Decimal(str(mean))), 
                                    '{:.3E}'.format(Decimal(str(std)))]) \
                                        for mean, std in zip(metric_means, metric_stds)]
            table_dict[metric] = metric_str

        df = pd.DataFrame(data=table_dict)
        return df