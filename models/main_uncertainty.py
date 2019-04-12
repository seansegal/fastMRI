"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time
import copy
import datetime
import os
import yaml

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.args import Args
from common.evaluate import METRIC_FUNCS, VolumeUncertaintyMetrics
from data import transforms
from data.mri_data import SliceData, MultiSliceData
from data.volume_sampler import VolumeSampler
from data.same_size_sampler import SameSizeSampler
from torch.utils.data.dataloader import default_collate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset(args, transform, split):
    if args.model == 'unet_volumes':
        return MultiSliceData(
            root=args.data_path / f'{args.challenge}_{split}',
            transform=transform,
            sample_rate=args.sample_rate,
            overfit=args.overfit,
            challenge=args.challenge,
            num_volumes=args.num_volumes
        )
    return SliceData(
        root=args.data_path / f'{args.challenge}_{split}',
        transform=transform,
        sample_rate=args.sample_rate,
        overfit=args.overfit,
        challenge=args.challenge,
    )


def create_datasets(args, mri_model):
    train_transform, val_transform, _ = mri_model.get_transforms(args)

    train_data = create_dataset(args, train_transform, 'train')
    if args.overfit:
        return train_data, train_data
    dev_data = create_dataset(args, train_transform, 'val')
    return dev_data, train_data


def get_batch_sampler(dataset, batch_size, display=False):
    return SameSizeSampler(dataset, batch_size, display)


def create_data_loaders(args, mri_model):
    dev_data, train_data = create_datasets(args, mri_model)

    try:
        train_batch_sampler = mri_model.get_batch_sampler(train_data, batch_size=args.batch_size)
    except AttributeError:
        train_batch_sampler = None
    try:
        collate_fn = mri_model.get_collate_fn()
    except AttributeError:
        collate_fn = default_collate

    display_data = copy.deepcopy(dev_data)
    display_batch_sampler = get_batch_sampler(display_data, batch_size=16, display=True)

    if train_batch_sampler is None:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset=train_data,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_sampler=VolumeSampler(dev_data, batches_per_volume=args.batches_per_volume),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if display_batch_sampler is None:
        display_loader = DataLoader(
            dataset=display_data,
            batch_size=16,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        display_loader = DataLoader(
            dataset=display_data,
            batch_sampler=display_batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return train_loader, dev_loader, display_loader


def plot_confidences_vs_l1_loss(reconstruction, target, confidence, save_path):
    loss_matrix = F.l1_loss(reconstruction, target, reduction='none')
    batch_losses = loss_matrix.sum(dim=2).sum(dim=1)
    confidence = confidence.detach().cpu().numpy()
    batch_losses = batch_losses.detach().cpu().numpy()
    conf_order = np.flip(confidence.argsort())
    losses = []
    for idx in conf_order:
        losses.append(batch_losses[idx])
    plt.plot(losses, '*')
    plt.xlabel('Confidence Rank')
    plt.ylabel('L1 Loss')
    plt.savefig(save_path)
    plt.clf()


def train_epoch(args, epoch, model, train_step, data_loader, optimizer, writer, inf):
    model.train()
    avg_loss = 0.
    start_epoch = end_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        loss = train_step(model, data, device=args.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('Train/TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - end_iter:.4f}s '
            )
        end_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def wrap_confidence(inference_output):
    if len(inference_output) == 2:  # inference() returns output_dict, target
        return inference_output[0], inference_output[1], torch.zeros(inference_output[0].size(0)), None
    elif len(inference_output) == 3:
        return inference_output[0], inference_output[1], inference_output[2], None
    elif len(inference_output) == 4:
        return inference_output


def write_metrics_to_tb(metrics, writer, epoch, name):
    means = metrics.top_k(k=1.0)
    for metric, mean in means.items():
        writer.add_scalar('Train/{}_Mean_{}'.format(name, metric), mean, epoch)
    ks = [0.25, 0.5, 0.75]
    for k in ks:
        means = metrics.top_k(k=k)
        for metric, mean in means.items():
            writer.add_scalar('Train/K_{}_{}_Mean_{}'.format(k, name, metric), mean, epoch)


def create_volume_dict(target, output, slice_confidences, sigmas, input, model_name, output_name, volume_id):
    return dict(
        target=target,
        output=output,
        slice_confidences=slice_confidences,
        sigmas=sigmas,
        input=input,
        model_name=model_name,
        output_name=output_name,  # Will always be 'model' (from old hack when we used keys for different model outputs)
        volume_id=volume_id
    )


def evaluate(
        device,
        model_name,
        model,
        inference,
        data_loader,
        batches_per_volume=1,
        epoch=0,
        writer=None,
        eval_callbacks=[]):
    start = time.perf_counter()
    metrics = None
    batch_per_volume = 0
    output_volume, target_volume = {}, {}
    confidence_volume = {}
    input_volume = {}
    sigmas_volume = {}
    volume_id = 0
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            output_dict, target, confidence, sigmas = wrap_confidence(inference(model, data, device=device))

#            if iter % 10 == 0: # TODO(Sean): Move to a callback
#                # Save one in every 10 confidence vs.
#                save_path = os.path.join(confidence_slice_dir, 'iter_{}.png'.format(iter))
#                plot_confidences_vs_l1_loss(output_dict, target, confidence, save_path)

            # HACK for now to let model output multiple predictions and get logging
            # for each prediction indepdently.
            if not isinstance(output_dict, dict):
                output_dict = {'model': output_dict}

            if metrics is None:
                metrics = {k: VolumeUncertaintyMetrics(METRIC_FUNCS) for k in output_dict}

            for k, output in output_dict.items():

                if batch_per_volume == 0:
                    target_volume[k] = target.cpu().numpy()
                    output_volume[k] = output.cpu().numpy()
                    confidence_volume[k] = confidence.cpu().numpy()
                    input_volume[k] = data[0].cpu().numpy()
                    if sigmas is not None:
                        sigmas_volume[k] = sigmas.cpu().numpy()
                    else:
                        sigmas_volume[k] = None
                else:
                    target_volume[k] = np.concatenate([target_volume[k], target.cpu().numpy()], axis=0)
                    output_volume[k] = np.concatenate([output_volume[k], output.cpu().numpy()], axis=0)
                    confidence_volume[k] = np.concatenate([confidence_volume[k], confidence.cpu().numpy()], axis=0)
                    input_volume[k] = np.concatenate([input_volume[k], data[0].cpu().numpy()], axis=0)
                    if sigmas is not None:
                        sigmas_volume[k] = np.concatenate([sigmas_volume[k], sigmas.cpu().numpy()], axis=0)

                batch_per_volume += 1
                if batch_per_volume == batches_per_volume:
                    metrics[k].push(target_volume[k], output_volume[k], np.sum(confidence_volume[k]))
                    volume_dict = create_volume_dict(
                        target_volume[k],
                        output_volume[k],
                        confidence_volume[k],
                        sigmas_volume[k],
                        input_volume[k],
                        model_name,
                        k,
                        volume_id)
                    for cb in eval_callbacks:
                        cb(volume_dict)
                    volume_id += 1
                    batch_per_volume = 0

    if writer is not None:
        for output_key in output_dict.keys():
            write_metrics_to_tb(metrics[output_key], writer, epoch, output_key)
    return metrics['model'].top_k(k=1.0)['NMSE'], time.perf_counter() - start


def visualize(args, epoch, model, inference, data_loader, writer):

    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def overlay_uncertainty(image, uncertainty, tag):
        image -= image.min()
        image /= image.max()
        uncertainty -= uncertainty.min()
        uncertainty /= uncertainty.max()
        # Convert to RGB
        image = image.expand(-1, 3, -1, -1)
        uncertainty = torch.cat([torch.zeros_like(uncertainty), uncertainty, uncertainty], 1)
        # Overlay
        image = image * 0.7 + uncertainty * 0.3
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            output, target, _, sigmas = wrap_confidence(inference(model, data, device=args.device))
            # HACK to make images look good in tensorboard
            output, mean_o, std_o = transforms.normalize_instance(output)
            output = output.clamp(-6, 6)
            output = transforms.unnormalize(output, mean_o, std_o)
            target, mean_t, std_t = transforms.normalize_instance(target)
            target = target.clamp(-6, 6)
            target = transforms.unnormalize(target, mean_t, std_t)

            output = output.unsqueeze(1)  # [batch_sz, h, w] --> [batch_sz, 1, h, w]
            target = target.unsqueeze(1)  # [batch_sz, h, w] --> [batch_sz, 1, h, w]
            error = torch.abs(target - output)

            if sigmas is not None:
                sigmas, mean_s, std_s = transforms.normalize_instance(sigmas)
                sigmas = sigmas.clamp(-6, 6)
                sigmas = transforms.unnormalize(sigmas, mean_s, std_s)
                sigmas = sigmas.unsqueeze(1)  # [batch_sz, h, w] --> [batch_sz, 1, h, w]

            if isinstance(output, dict):
                for k, output_val in output.items():
                    # save_image(input, 'Input_{}'.format(k))
                    save_image(target, 'Target_{}'.format(k))
                    save_image(output, 'Reconstruction_{}'.format(k))
                    save_image(error, 'Error_{}'.format(k))
                    save_image(sigmas, 'Std_{}'.format(k))
                    if sigmas is not None:
                        overlay_uncertainty(error, sigmas, 'Overlay_Error_Std_{}'.format(k))
                        overlay_uncertainty(output, sigmas, 'Overlay_Reconstruction_Std_{}'.format(k))
            else:
                # save_image(input, 'Input')
                save_image(target, 'Target')
                save_image(output, 'Reconstruction')
                save_image(error, 'Error')
                if sigmas is not None:
                    save_image(sigmas, 'Std')
                    overlay_uncertainty(error, sigmas, 'Overlay_Error_Std')
                    overlay_uncertainty(output, sigmas, 'Overlay_Reconstruction_Std')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def load_model(checkpoint_file, mri_model):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = mri_model.build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = mri_model.build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def load_model_config(config_file):
    if config_file:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return None


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    module = __import__('models', fromlist=[args.model])
    mri_model = getattr(
        module, args.model)(load_model_config(args.model_config))

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint, mri_model)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    elif args.checkpoint:
        # Don't overwrite args (used so that we can just evaluate the model)
        checkpoint, model, optimizer = load_model(args.checkpoint, mri_model)
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = mri_model.build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = mri_model.build_optim(args, model)
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args, mri_model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    train_step_func = mri_model.train_step
    inference_func = mri_model.inference

    if args.evaluate_only:
        dev_loss, dev_time = evaluate(args.device, args.model, model, inference_func,
                                      dev_loader, batches_per_volume=args.batches_per_volume, epoch=0, writer=writer)
        writer.close()
        return

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == 0:
            evaluate(
                args.device,
                args.model,
                model,
                inference_func,
                dev_loader,
                batches_per_volume=args.batches_per_volume,
                epoch=-1,
                writer=writer)
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_step_func,
                                             train_loader, optimizer, writer, inference_func)
        dev_loss, dev_time = evaluate(args.device, args.model, model, inference_func,
                                      dev_loader, batches_per_volume=args.batches_per_volume, epoch=epoch, writer=writer)
        if epoch % 1 == 0:
            visualize(args, epoch, model, inference_func, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--overfit', action='store_true',
                        help='If set, it will use the same dataset for training and val')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of PyTorch workers')
    parser.add_argument('--batches-per-volume', type=int, default=1,
                        help='Number of batches to break up volume into when evaluting. Set to higher if you run OOM.')
    parser.add_argument('--model', type=str, required=True, help='Model directory.')
    parser.add_argument('--model-config', type=str, default=None, help='model config, if applicable')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--exp-dir', default='experiments/uncertainty', type=str, help='tensorboard and checkpoint dir')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    if args.data_path is None:
        if os.path.isdir("/localssd/fastMRI"):
            args.data_path = pathlib.Path("/localssd/fastMRI")
        elif os.path.isdir("/local/datasets/fastMRI"):
            args.data_path = pathlib.Path("/local/datasets/fastMRI")
        else:
            raise ValueError('Need to provide a valid path for the dataset')
    now = datetime.datetime.now()
    exp_name = args.exp_name + '_' + now.isoformat()
    args.exp_dir = pathlib.Path(os.path.join(args.exp_dir, exp_name))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
