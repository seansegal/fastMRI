import matplotlib.pyplot as plt
from common.evaluate import SliceUncertaintyMetrics, METRIC_FUNCS


class PlotTopKBySlice():

    def __init__(self):
        self.uncertainty_metrics = SliceUncertaintyMetrics(METRIC_FUNCS)

    def __call__(self, volume_dict, model_name=None):
        """
        Args:
            volume_dict (dict):             Dictionary with keys:
                                                - target (numpy.ndarray) Shape: (num_slices, height, width)
                                                - output (numpy.ndarray) Shape: (num_slices, height, width)
                                                - slice_confidences (numpy.ndarray) Shape: (num_slices, )
        """
        model_name = model_name if model_name is not None else volume_dict['model_name']
        self.uncertainty_metrics.push(
            volume_dict['target'],
            volume_dict['output'],
            volume_dict['slice_confidences'],
            model_name)

    def produce_plot_data(self, compute_baseline=True):
        return self.uncertainty_metrics._plot_means_vs_k(compute_baseline=compute_baseline)

    def produce_plot(self, data_list):
        figures = []
        for metric in data_list[0][1]:
            fig = plt.figure()
            legend = []
            for data in data_list:
                model_name, metric_unc_means, baseline_means, k_vals = data
                plt.plot(k_vals, metric_unc_means[metric])
                legend.append(model_name)
                if len(baseline_means[metric]) > 0:
                    plt.plot(k_vals, baseline_means[metric])
                    legend.append('Random Choice Baseline')
            plt.xlabel('K')
            plt.ylabel(metric)
            plt.legend(legend)
            figures.append(fig)
        return figures
