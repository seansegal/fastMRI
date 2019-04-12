import logging
import pathlib
import random
import shutil
import time
import datetime
import itertools

import numpy as np
import torch
import cv2
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import MaskFunc
from data import transforms
from data.mri_data import SliceData
from data.dicom_data import SliceDICOM, DICOM_collate
from models.gan.gan_model import GanGenerator, GanDiscriminator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = transforms.get_mask(kspace, self.mask_func, seed)
        masked_kspace = kspace * mask
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = transforms.to_tensor(target)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return image, target, mean, std, attrs['norm'].astype(np.float32)

class DICOMTransform:
    def __init__(self, resolution):
        """
        Args:
            resolution (int): Resolution of the image.
        """
        self.resolution = resolution

    def __call__(self, image):
        """
        Args:
            image (numpy.array): DICOM image
        Returns:
            image (torch.Tensor): Zero-filled input image.
        """

        # image = np.rot90(image, axes=(0, 1)).copy()
        image = np.flip(image, 0)

        # if image.shape[0] < self.resolution or image.shape[1] < self.resolution:
        #     return None
        # # Crop center
        # image = transforms.center_crop(image, (self.resolution, self.resolution))

        res_crop = min(image.shape[0], image.shape[1])
        image = transforms.center_crop(image, (res_crop, res_crop))
        image = cv2.resize(image, dsize=(self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
        
        # Normalize input
        image = transforms.to_tensor(image)
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        return image  

def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.data_path / f'{args.challenge}_train',
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )
    if not args.overfit:
        dev_data = SliceData(
            root=args.data_path / f'{args.challenge}_val',
            transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
        )
    else:
        dev_data = SliceData(
            root=args.data_path / f'{args.challenge}_train',
            transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
        )
    if args.use_dicom:
        dicom_data = SliceDICOM(root=args.data_path, 
                                transform=DICOMTransform(args.resolution),
                                sample_rate=args.sample_rate,
        )
        return dev_data, train_data, dicom_data
    return dev_data, train_data


def create_data_loaders(args):
    if not args.use_dicom:
        dev_data, train_data = create_datasets(args)
    else:
        dev_data, train_data, dicom_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.use_dicom:
        dicom_loader = DataLoader(
            dataset=dicom_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=DICOM_collate,
    )
        return train_loader, dev_loader, display_loader, dicom_loader
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, discriminator, generator, data_loader, optimizerD, optimizerG, writer, dicom_loader=None):
    discriminator.train()
    generator.train()
    start_epoch = end_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    if args.use_dicom:
        dicom_iter = iter(dicom_loader)
    for i, data in enumerate(data_loader):
        image_u, image_f, mean, std, norm = data
        image_u = image_u.unsqueeze(1).to(args.device)
        image_f = image_f.to(args.device)
        image_f_hat = generator(image_u)

        ### Optimize discriminator
        if epoch >= args.start_gan:
            p_image_f = discriminator(image_f.unsqueeze(1).to(args.device))
            real_loss = F.binary_cross_entropy_with_logits(p_image_f, torch.ones(p_image_f.shape).cuda())
            writer.add_scalar('Debug/RealProb', F.sigmoid(p_image_f).mean().item(), global_step + i)
            p_image_f_hat = discriminator(image_f_hat.detach())
            fake_loss = F.binary_cross_entropy_with_logits(p_image_f_hat, torch.zeros(p_image_f_hat.shape).cuda())
            writer.add_scalar('Debug/FakeProb', F.sigmoid(p_image_f_hat).mean().item(), global_step + i)
            if args.use_dicom:
                dicom_image = None
                while dicom_image is None:
                    try:
                        dicom_image = next(dicom_iter)
                    except StopIteration:
                        dicom_iter = iter(dicom_loader)
                p_dicom = discriminator(dicom_image.unsqueeze(1).to(args.device))
                dicom_loss = F.binary_cross_entropy_with_logits(p_dicom, torch.zeros(p_dicom.shape).cuda())
                writer.add_scalar('Debug/DicomProb', F.sigmoid(p_dicom).mean().item(), global_step + i)
                lossD = real_loss + fake_loss + dicom_loss
                writer.add_scalar('Train/DicomLoss', dicom_loss.item(), global_step + i)
            else:
                lossD = real_loss + fake_loss
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            writer.add_scalar('Train/RealLoss', real_loss.item(), global_step + i)
            writer.add_scalar('Train/FakeLoss', fake_loss.item(), global_step + i)
            writer.add_scalar('Train/TrainLossD', lossD.item(), global_step + i)

        ### Optimize generator
        consistency_loss = 10.0 * F.l1_loss(image_f_hat.squeeze(1), image_f)
        if epoch >= args.start_gan:
            p_image_f_hat = discriminator(image_f_hat)
            # disc_loss = min(float(global_step + i) / 10000, 1.0) * F.binary_cross_entropy_with_logits(
            #                                                         p_image_f_hat, torch.ones(p_image_f_hat.shape).cuda())
            disc_loss = F.binary_cross_entropy_with_logits(p_image_f_hat, torch.ones(p_image_f_hat.shape).cuda())
            lossG =  disc_loss + consistency_loss
            writer.add_scalar('Train/DiscLossG', disc_loss.item(), global_step + i)
        else:
            lossG = consistency_loss
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

        writer.add_scalar('Train/ConsistencyLoss', consistency_loss.item(), global_step + i)
        writer.add_scalar('Train/TrainLossG', lossG.item(), global_step + i)
    
        avg_loss = 0.99 * avg_loss + 0.01 * consistency_loss.item() if i > 0 else consistency_loss.item()

        if i % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{i:4d}/{len(data_loader):4d}] '
                f'Loss = {consistency_loss.item():.4f} Avg Loss = {avg_loss:.4f} '
                f'Time = {time.perf_counter() - end_iter:.4f}s '
            )
        end_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, generator, data_loader, writer):
    generator.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            image_u, image_f, mean, std, norm = data
            image_u = image_u.unsqueeze(1).to(args.device)
            image_f = image_f.to(args.device)
            image_f_hat = generator(image_u).squeeze(1)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            image_f = image_f * std + mean
            image_f_hat = image_f_hat * std + mean

            norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            loss = F.mse_loss(image_f_hat / norm, image_f / norm, size_average=False) / args.batch_size
            losses.append(loss.item())
        writer.add_scalar('Eval/Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, generator, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    generator.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = generator(input)
            save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, discriminator, generator, optimizerD, optimizerG, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    gen = GanGenerator(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    disc = GanDiscriminator(
        in_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools+1,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return disc, gen


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    discriminator, generator = build_model(args)
    if args.data_parallel:
        discriminator = torch.nn.DataParallel(discriminator)
        generator = torch.nn.DataParallel(generator)
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator.load_state_dict(checkpoint['generator'])

    optimizerD = build_optim(args, discriminator.parameters())
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    optimizerG = build_optim(args, generator.parameters())
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    return checkpoint, discriminator, generator, optimizerD, optimizerG


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, discriminator, generator, optimizerD, optimizerG = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        discriminator, generator = build_model(args)
        if args.data_parallel:
            discriminator = torch.nn.DataParallel(discriminator)
            generator = torch.nn.DataParallel(generator)
        optimizerD = build_optim(args, discriminator.parameters())
        optimizerG = build_optim(args, generator.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(discriminator)
    logging.info(generator)

    if args.use_dicom:
        train_loader, dev_loader, display_loader, dicom_loader = create_data_loaders(args)
    else:
        train_loader, dev_loader, display_loader = create_data_loaders(args)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, args.lr_step_size, args.lr_gamma)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        schedulerD.step(epoch)
        schedulerG.step(epoch)
        if args.use_dicom:
            train_loss, train_time = train_epoch(args, epoch, discriminator, generator, train_loader, 
                                                    optimizerD, optimizerG, writer, dicom_loader)
        else:
            train_loss, train_time = train_epoch(args, epoch, discriminator, generator, train_loader, 
                                                    optimizerD, optimizerG, writer)
        dev_loss, dev_time = evaluate(args, epoch, generator, dev_loader, writer)
        if not args.overfit or epoch % 10 == 0:
            visualize(args, epoch, generator, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, discriminator, generator, optimizerD, optimizerG, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] ConsistencyLoss = {train_loss:.4g}'
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
    parser.add_argument('--exp-name', type=str, default='gan',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--overfit', action='store_true',
                        help='If set, it will use the same dataset for training and val')
    parser.add_argument('--start-gan', type=int, default=0, help='Number of epochs of generator pretraining')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of PyTorch workers')
    parser.add_argument('--use-dicom', action='store_true', help='Use DICOM images as fake reconstruction')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    now = datetime.datetime.now()
    exp_name = args.exp_name + '_' + now.isoformat()
    args.exp_dir = pathlib.Path('experiments/' + exp_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
