from torch.utils.data.dataloader import default_collate
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import logging
import pathlib
import random
import shutil
import time
import copy
import datetime

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.args import Args
from common.tensorboard import write_metrics_to_tb
from common.evaluate import Metrics, METRIC_FUNCS
from common.subsample import MaskFunc
from data import transforms
from data.mri_data import SliceData, MultiSliceData
from models.unet_complex.unet_model import UnetModel
from data.volume_sampler import VolumeSampler
from data.same_size_sampler import SameSizeSampler


from torch.utils.data.dataloader import default_collate

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
def create_datasets(args):
    module = __import__('models.{}.train'.format(args.model), fromlist=[''])
    train_transform, val_transform = module.get_transforms(args)

    train_data = create_dataset(args, train_transform, 'train')
    if args.overfit: 
        return train_data, train_data
    dev_data = create_dataset(args, train_transform, 'val')
    return dev_data, train_data


def get_batch_sampler(dataset, batch_size, display=False):
    return SameSizeSampler(dataset, batch_size, display)
    

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

    module = __import__('models.{}.train'.format(args.model), fromlist=[''])
    try:
        train_batch_sampler = module.get_batch_sampler(train_data, batch_size=args.batch_size)
    except AttributeError:
        train_batch_sampler = None
    try:
        collate_fn = module.get_collate_fn()
    except AttributeError:
        collate_fn = default_collate

    display_data = copy.deepcopy(dev_data)
    display_batch_sampler = get_batch_sampler(display_data, batch_size=16, display=True)

    # try:
    #     display_data = copy.deepcopy(dev_data)
    #     display_batch_sampler = module.get_batch_sampler(display_data, batch_size=16, display=True)
    # except AttributeError:
    #     display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]
    #     display_batch_sampler = None

    if train_batch_sampler is None:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size = args.batch_size,
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


def train_epoch(args, epoch, generator, discriminator, train_step_generator, 
                train_step_discriminator, data_loader, optimizerG, optimizerD, writer):
    generator.train()
    discriminator.train()
    avg_loss = 0.
    start_epoch = end_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):

        # Optimize discriminator
        lossD_real, lossD_fake, p_target, p_output = train_step_discriminator(generator, discriminator, data, device=args.device)
        optimizerD.zero_grad()
        lossD_real.backward()
        lossD_fake.backward()
        optimizerD.step()
        for p in discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)

        # Optimize generator
        lossG, loss = train_step_generator(generator, discriminator, data, device=args.device)
        optimizerG.zero_grad()
        (lossG + loss).backward()
        optimizerG.step()

        # print(f'G:{lossG.item():.03f}, C:{loss.item():.03f}, Dreal:{lossD_real.item():.03f}, Dfake:{lossD_fake.item():.03f}')
        
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('Train/TrainLoss', loss.item(), global_step + iter)
        writer.add_scalar('TrainGan/ProbTarget', p_target.mean().item(), global_step + iter)
        writer.add_scalar('TrainGan/ProbOutput', p_output.mean().item(), global_step + iter)
        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - end_iter:.4f}s '
            )
        end_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, generator, inference, data_loader, writer):
    losses = []
    losses_consistent = []
    start = time.perf_counter()
    metrics = None
    batch_per_volume = 0
    output_volume, target_volume = {}, {}
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            output_dict, target = inference(generator, data, device=args.device)

            # HACK for now to let model output multiple predictions and get logging
            # for each prediction indepdently.
            if not isinstance(output_dict, dict):
                output_dict = {'model': output_dict}

            if metrics is None:
                metrics = {key: Metrics(METRIC_FUNCS) for key in output_dict.keys()}

            for k, output in output_dict.items():

                if batch_per_volume == 0:
                    target_volume[k] = target.cpu().numpy()
                    output_volume[k] = output.cpu().numpy()
                else:
                    target_volume[k] = np.concatenate([target_volume[k], target.cpu().numpy()], axis=0)
                    output_volume[k] = np.concatenate([output_volume[k], output.cpu().numpy()], axis=0)

                batch_per_volume += 1
                if batch_per_volume == args.batches_per_volume:
                    metrics[k].push(target_volume[k], output_volume[k])
                    batch_per_volume = 0

    for output_key in output_dict.keys():
        write_metrics_to_tb(metrics[output_key], writer, epoch, output_key)
    return metrics['model'].means()['NMSE'], time.perf_counter() - start


def visualize(args, epoch, generator, inference, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    generator.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            output, target = inference(generator, data, device=args.device)

            # HACK to make images look good in tensorboard
            output, mean_o, std_o = transforms.normalize_instance(output)
            output = output.clamp(-6, 6)
            output = transforms.unnormalize(output, mean_o, std_o)
            target, mean_t, std_t = transforms.normalize_instance(target)
            target = target.clamp(-6, 6)
            target = transforms.unnormalize(target, mean_t, std_t)

            output = output.unsqueeze(1) # [batch_sz, h, w] --> [batch_sz, 1, h, w]
            target = target.unsqueeze(1) # [batch_sz, h, w] --> [batch_sz, 1, h, w]
            if isinstance(output, dict):
                for k, output_val in output.items():
                    #save_image(input, 'Input_{}'.format(k))
                    save_image(target, 'Target_{}'.format(k))
                    save_image(output, 'Reconstruction_{}'.format(k))
                    save_image(torch.abs(target - output), 'Error_{}'.format(k))
            else:
                #save_image(input, 'Input')
                save_image(target, 'Target')
                save_image(output, 'Reconstruction')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, generator, discriminator, optimizerG, optimizerD, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'generator': generator.module.state_dict() if args.data_parallel else generator.state_dict(),
            'discriminator': discriminator.module.state_dict() if args.data_parallel else discriminator.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    module = __import__('models.{}.train'.format(args.model), fromlist=[''])
    return module.build_model(args)


def load_model(args, only_gen):
    checkpoint = torch.load(args.checkpoint)
    generator, discriminator = build_model(args)
    
    if only_gen:
        generator.load_state_dict(checkpoint['model'])
    else:
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    optimizerG, optimizerD = build_optim(args, generator.parameters(), discriminator.parameters())
    if only_gen:
        optimizerG.load_state_dict(checkpoint['optimizer'])
    else:
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
    return checkpoint, generator, discriminator, optimizerG, optimizerD


def build_optim(args, paramsG, paramsD):
    module = __import__('models.{}.train'.format(args.model), fromlist=[''])
    return module.build_optim(args, paramsG, paramsD)


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    assert not (args.resume and args.pretrained_generator) 

    if args.resume or args.pretrained_generator:
        checkpoint, generator, discriminator, optimizerG, optimizerD = load_model(args, only_gen=args.pretrained_generator)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        generator, discriminator = build_model(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        optimizerG, optimizerD = build_optim(args, generator.parameters(), discriminator.parameters())

        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, args.lr_step_size, args.lr_gamma)
    module = __import__('models.{}.train'.format(args.model), fromlist=[''])
    train_step_gen_func = module.train_step_generator
    train_step_disc_func = module.train_step_discriminator
    inference_func = module.inference

    for epoch in range(start_epoch, args.num_epochs):
        schedulerG.step(epoch)
        schedulerD.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, generator, discriminator, train_step_gen_func, 
                                            train_step_disc_func, train_loader, optimizerG, optimizerD, writer)
        dev_loss, dev_time = evaluate(args, epoch, generator, inference_func, dev_loader, writer)
        if epoch % 1 == 0:
            visualize(args, epoch, generator, inference_func, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, generator, discriminator, optimizerG, optimizerD, best_dev_loss, is_new_best)
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
    parser.add_argument('--pretrained-generator', action='store_true',
                        help='If set, resume the generator from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--overfit', action='store_true',
                        help='If set, it will use the same dataset for training and val')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of PyTorch workers')
    parser.add_argument('--batches-per-volume', type=int, default=1,
                        help='Number of batches to break up volume into when evaluting. Set to higher if you run OOM.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model directory.')
    parser.add_argument('--num-volumes', type=int, default=3,
                        help='Number of input volumes - only relevant for model_volumes.')
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
    args.exp_dir = pathlib.Path('experiments/gan/' + exp_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
