import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import torchvision
from models.mri_model import MRIModel
import data.transforms as transforms
import pytorch_ssim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_regression_ccn_model(in_channels):
    model = torchvision.models.resnet18(pretrained=False)

    # Predict a single value
    model.fc = nn.Linear(in_features=512, out_features=1)

    # 3 channels to in_channels
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


class GroupedModels():

    def __init__(self, model, unc_model):
        self.r_model = model
        self.unc_model = unc_model

    def train(self):
        self.r_model.eval()  # Assume r_model is always doing inference
        self.unc_model.train()

    def eval(self):
        self.r_model.eval()
        self.unc_model.eval()

    def load_state_dict(self, state_dict):
        return self.unc_model.load_state_dict(state_dict)

    def state_dict(self):
        return self.unc_model.state_dict()


class DataTransformWrapper():
    def __init__(self, data_transform):
        self.data_transform = data_transform

    def __call__(self, kspace, target, attrs, fname, slice):
        return self.data_transform(kspace, target, attrs, fname, slice) + (attrs['max'].astype(np.float32),)


class LossPredictionModel(MRIModel):
    """ Wraps a model with no confidence scores and predicts loss.
    """

    def __init__(self, model_config):
        assert model_config is not None
        self.module = __import__(model_config['model_module'], fromlist=[''])
        self.trained_checkpoint = model_config['checkpoint']
        self.loss = model_config['loss'] if 'loss' in model_config else 'l1'

    def _load_trained_model(self):
        checkpoint = torch.load(self.trained_checkpoint)
        args = checkpoint['args']
        model = self.module.build_model(args)
        model.load_state_dict(checkpoint['model'])
        return model

    def get_transforms(self, args):
        train_transform, val_transform, test_transform = self.module.get_transforms(args)
        return DataTransformWrapper(train_transform), val_transform, test_transform

    def train_step(self, model, data, device):
        model.unc_model.train()
        model.r_model.eval()
        input, target, mean, std, norm, _, data_range = data
        with torch.no_grad():
            output, target = self.module.inference(model.r_model, data[:-1], device)
            if self.loss == 'l1':
                target_loss = F.l1_loss(output, target, reduction='none').sum(dim=2).sum(dim=1).unsqueeze(1)
            elif self.loss == 'ssim':
                SSIMLoss = pytorch_ssim.SSIM(window_size=7, size_average=False)
                target_loss = SSIMLoss(output.unsqueeze(1), target.unsqueeze(1), data_range.to(device)).unsqueeze(1)
                # Normalize target loss
                target_loss = (target_loss - 0.9) / 0.1
            else:
                ValueError('Invalid loss')
        mean = mean.unsqueeze(1).unsqueeze(2).to(device)
        std = std.unsqueeze(1).unsqueeze(2).to(device)
        output = transforms.normalize(output, mean, std, eps=1e-11)
        loss_prediction = model.unc_model(output.unsqueeze(1))
        return F.mse_loss(loss_prediction, target_loss)

    def inference(self, model, data, device):
        model.unc_model.eval()
        model.r_model.eval()
        input, _, mean, std, _, target, _ = data
        with torch.no_grad():
            output, target = self.module.inference(model.r_model, data[:-1], device)
        # Renormalize
        mean = mean.unsqueeze(1).unsqueeze(2).to(device)
        std = std.unsqueeze(1).unsqueeze(2).to(device)
        output_normalized = transforms.normalize(output, mean, std, eps=1e-11)
        loss_prediction = model.unc_model(output_normalized.unsqueeze(1))

        if self.loss == 'l1':
            confidence = -1 * loss_prediction
        elif self.loss == 'ssim':
            confidence = loss_prediction
        else:
            raise ValueError('Invalid loss')

        return output, target, confidence.squeeze(1)

    def build_model(self, args):
        original_model = self._load_trained_model().to(args.device)
        confidence_model = create_regression_ccn_model(in_channels=1).to(args.device)
        return GroupedModels(original_model, confidence_model)

    def build_optim(self, args, model):
        optimizer = torch.optim.RMSprop(model.unc_model.parameters(), args.lr, weight_decay=args.weight_decay)
        return optimizer
