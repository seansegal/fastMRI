from skimage.measure import compare_psnr, compare_ssim
from runstats import Statistics
import matplotlib.pyplot as plt
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
import os
import pickle
from argparse import ArgumentParser

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


def norm(gt, pred):
    return np.linalg.norm(gt) ** 2


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def replace_least_confident(target_volume, prediction_volume, confidences, k=0.5, random_shuffle_baseline=False):
    prediction_volume = prediction_volume.copy()
    assert k > 0
    assert k <= 1.0
    num_to_replace = int((1 - k) * target_volume.shape[0])
    conf_order = confidences.argsort()  # Indexes in order least to most confident
    if random_shuffle_baseline:
        np.random.shuffle(conf_order)
    prediction_volume[conf_order[:num_to_replace]] = target_volume[conf_order[:num_to_replace]]
    return prediction_volume


class SliceUncertaintyMetrics:
    """ Metrics for models that output uncertainty at the slice level. The primary metric we use is
    top-K, in which we take the K% most confident slices and use the target for all other slices
    in each volume. Of course, this metric will be strictly better than the original metric (since
    we are replacing some reconstructions with the ground truth) so it must be compared to a method
    which randomly selects slices to keep.
    """

    def __init__(self, metric_funcs, k_vals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        self.metric_funcs = metric_funcs
        self.metrics = {}
        self.baseline = {
            metric: [[] for _ in k_vals] for metric in metric_funcs
        }
        self.k_vals = k_vals

    def push(self, target, recons, confidences, model_name):
        """ Pushes a target and reconstruction volume. Stores all information needed to compute
        top-K statistics over the dataset. confidences here is at the slice level, not the volume
        level.
        Args:
            target (numpy.ndarray):                  Shape: (num_slices, height, width)
            recons (numpy.ndarray):                  Shape: (num_slices, height, width)
            confidences (numpy.ndarray):             Shape: (num_slices, )
        """
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                metric: [[] for _ in self.k_vals] for metric in self.metric_funcs
            }
        for k_idx, k in enumerate(self.k_vals):
            recon_unc = replace_least_confident(target, recons, confidences, k=k)
            recons_base = replace_least_confident(target, recons, confidences, k=k, random_shuffle_baseline=True)
            for metric, func in self.metric_funcs.items():
                self.metrics[model_name][metric][k_idx].append(func(target, recon_unc))
                self.baseline[metric][k_idx].append(func(target, recons_base))

    def _plot_means_vs_k(self, compute_baseline=True):
        results = {}
        baselines = {}
        assert len(self.metrics) == 1
        model_name = list(self.metrics.keys())[0]
        for metric in self.metric_funcs:
            unc_means = []
            baseline_means = []
            for k_idx, k in enumerate(self.k_vals):
                unc_means.append(np.array(self.metrics[model_name][metric][k_idx]).mean())
                if compute_baseline:
                    baseline_means.append(np.array(self.baseline[metric][k_idx]).mean())
            results[metric] = unc_means
            baselines[metric] = baseline_means
        return model_name, results, baselines, self.k_vals

    def plot_means_vs_k(self, savedir=None):
        figures = []
        for metric in self.metric_funcs:
            unc_means = {model_name: list() for model_name in self.metrics}
            baseline_means = []
            for k_idx, k in enumerate(self.k_vals):
                for model_name in self.metrics:
                    unc_means[model_name].append(np.array(self.metrics[model_name][metric][k_idx]).mean())
                baseline_means.append(np.array(self.baseline[metric][k_idx]).mean())
            fig = plt.figure()
            for _, means in unc_means.items():
                plt.plot(self.k_vals, means)
            plt.plot(self.k_vals, baseline_means)
            plt.xlabel('K')
            plt.ylabel(metric)
            plt.legend(list(unc_means.keys()) + ['Random Choice Baseline'])
            if savedir is not None:
                savepath = os.path.join(savedir, '{}.png'.format(metric))
                plt.savefig(savepath)
            figures.append(fig)
        return figures


class VolumeUncertaintyMetrics:
    """
        Metrics with uncertainty. Uncertainty scores must be by VOLUME, since
        that is how the metrics are computed.
    """

    def __init__(self, metric_funcs):
        self.metric_funcs = metric_funcs
        self.metrics = {
            metric: [] for metric in metric_funcs
        }
        self.confidences = []

    def push(self, target, recons, confidence):
        for metric, func in self.metric_funcs.items():
            self.metrics[metric].append(func(target, recons))
        self.confidences.append(confidence)

    def top_k_metric(self, metric, k):
        """ Returns top_k score for a single metric.
        Args:
            metric (str):         Metric key
            k (float):            Value where 0 < k <= 1.0
        """
        if k <= 0 or k > 1:
            raise ValueError('Requires 0 < k <= 1')
        metrics = np.array(self.metrics[metric])
        confidences = np.array(self.confidences)
        confidence_order = np.flip(confidences.argsort())
        num_to_take = int(round(k * len(confidences)))
        return np.mean(metrics[confidence_order[:num_to_take]])

    def top_k(self, k):
        """ Returns top_k score for all metrics
        Args:
            k (float):          Value where 0 < k <= 1.0
        """
        results = {}
        for metric in self.metric_funcs:
            results[metric] = self.top_k_metric(metric, k)
        return results

    def plot_confidence_curve(self, save_dir):
        """ Creates a plot of confidence vs. each metric. """
        confidences = np.array(self.confidences)
        confidence_order = np.flip(confidences.argsort())
        for metric, values in self.metrics.items():
            stats = Statistics()
            cum_metrics = []
            for ex_idx in confidence_order:
                stats.push(values[ex_idx])
                cum_metrics.append(stats.mean())

            plt.plot(cum_metrics)
            plt.xlabel('Confidence Ranking')
            plt.ylabel(metric)
            save_pickle({'confidences': confidences, 'metrics': cum_metrics},
                        os.path.join(save_dir, '{}.p'.format(metric)))
            plt.savefig(os.path.join(save_dir, '{}.png'.format(metric)))
            plt.clf()


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


ARRAY_METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    NORM=norm,
)


class ArrayMetrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {}
        self.metric_funcs = metric_funcs

    def push(self, target, recons, model_name):
        if model_name not in self.metrics:
            self.metrics[model_name] = {metric: [] for metric in self.metric_funcs}
        for metric, func in self.metric_funcs.items():
            self.metrics[model_name][metric].append(func(target, recons))

    def scatter_metrics_norm(self, save_dir=None):
        figures = []
        metrics = {'NMSE': 1e2, 'MSE': 1e10}
        for m, v in metrics.items():
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            for model in self.metrics:
                plt.scatter(np.asarray(self.metrics[model]['NORM']), 
                            v * np.asarray(self.metrics[model][m]),
                            label=model)
                plt.xlabel('norm')
                plt.ylabel(m)
                # plt.legend(list(unc_means.keys()) + ['Random Choice Baseline'])
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, 'scatter_{}_norm.png'.format(m)))
            plt.legend()
            figures.append(fig)
        return figures



def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
                args.predictions_path / tgt_file.name) as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue
            target = target[recons_key].value
            recons = recons['reconstruction'].value
            metrics.push(target, recons)
    return metrics


def within(m, s, y, t, true):
    direct = ((m <= y + t*s)*(m >= y - t*s)).sum()
    scaled = ((m <= y + 0.2*t*s)*(m >= y - 0.2*t*s)).sum()
    return [direct, scaled, true*m.size]

def total(m, s, y):
    return m.size

CALIBRATION_METRIC_FUNCS = dict(
    WITHIN01=lambda m, s, y: within(m, s, y, 0.1, 0.08),
    WITHIN03=lambda m, s, y: within(m, s, y, 0.3, 0.24),
    WITHIN05=lambda m, s, y: within(m, s, y, 0.5, 0.39),
    WITHIN1=lambda m, s, y: within(m, s, y, 1.0, 0.69),
    WITHIN20=lambda m, s, y: within(m, s, y, 2.0, 0.95),
    WITHIN30=lambda m, s, y: within(m, s, y, 3.0, 0.99),
    TOTAL=total
)

class CalibrationMetrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {}
        self.metric_funcs = metric_funcs

    def push(self, target, recons, sigmas, model_name):
        if model_name not in self.metrics:
            self.metrics[model_name] = {metric: [] for metric in self.metric_funcs}
        for metric, func in self.metric_funcs.items():
            self.metrics[model_name][metric].append(func(recons, sigmas, target))

    def print_calibration(self):
        for m in CALIBRATION_METRIC_FUNCS:
            for model in self.metrics:
                if m != 'TOTAL':
                    d, s, t = list(zip(*self.metrics[model][m]))
                    d = sum(d) / float(sum(self.metrics[model]['TOTAL']))
                    s = sum(s) / float(sum(self.metrics[model]['TOTAL']))
                    t = sum(t) / float(sum(self.metrics[model]['TOTAL']))
                    print(m, model, d, s, t)

    def barplot_calibration(self):
        for model in self.metrics:
            ds, ss, ts = [], [], []
            for m in CALIBRATION_METRIC_FUNCS:
                if m != 'TOTAL':
                    d, s, t = list(zip(*self.metrics[model][m]))
                    ds.append(sum(d) / float(sum(self.metrics[model]['TOTAL'])))
                    ss.append(sum(s) / float(sum(self.metrics[model]['TOTAL'])))
                    ts.append(sum(t) / float(sum(self.metrics[model]['TOTAL'])))
            
            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(11)
            num_stds = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
            index = np.arange(len(num_stds))
            bar_width = 0.25
            opacity=0.4
            rects1 = ax.bar(index, ds, bar_width,
                alpha=opacity, color='blue', label='Output')
            rects2 = ax.bar(index + bar_width, ss, bar_width,
                alpha=opacity, color='orange', label='Scaled')
            rects3 = ax.bar(index + 2*bar_width, ts, bar_width,
                alpha=opacity, color='green', label='True')
            ax.set_xlabel('Number of standard deviations')
            ax.set_ylabel('Percentage of examples within x*stddev')
            ax.set_title('Calibration plot')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
            ax.legend()

            fig.tight_layout()
            break
        return fig
            



def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
                args.predictions_path / tgt_file.name) as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue
            target = target[recons_key].value
            recons = recons['reconstruction'].value
            metrics.push(target, recons)
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                        help='Which challenge')
    parser.add_argument('--acquisition', choices=['CORPD_FBK', 'CORPDFS_FBK'], default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')
    args = parser.parse_args()

    recons_key = 'reconstruction_rss' if args.challenge == 'multicoil' else 'reconstruction_esc'
    metrics = evaluate(args, recons_key)
    print(metrics)
