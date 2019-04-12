"""
    Authors: Sean, Sergio & Siva.
"""

from abc import ABC, abstractmethod

class MRIModel(ABC):
    """ 
        Abstract base class for a fastMRI model.
    """

    @abstractmethod
    def get_transforms(self, args):
        """ Defines the data transforms on the SliceData dataset.
        """
        pass

    @abstractmethod
    def train_step(self, model, data, device):
        """ Defines one training step iteration. 
        Returns:
            torch.Tensor:               A scalar loss value. Should be able to call loss.backward()
        """
        pass

    @abstractmethod
    def inference(self, model, data, device):
        """ Defines model at inference
        """
        pass

    @abstractmethod
    def build_model(self, args):
        """ Defines the neural network model. 
        Returns:
            torch.nn.Module:                The neural network model.
        """
        pass

    @abstractmethod
    def build_optim(self, args, model):
        """ Defines the optimizer
        Returns:
            torch.optim.Optimizer:         Optimizer
        """
        pass
