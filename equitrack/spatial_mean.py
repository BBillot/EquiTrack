import torch
import numpy as np
from torch import nn
from equitrack.utils import add_axis


class SpatialMeans(nn.Module):

    def __init__(self, input_shape, return_power=True, **kwargs):
        """This function takes in a torch tensor of shape [B, C, H, W, D], and returns the center of mass of
        each channel [B, C, 3]. It assumes the input has non-negative values.
        :param input_shape: list representing the shape of inputs [B, C, H, W, D]
        :param return_power: whether to return an additional output of size [B, C, 1] indicating the sum
        of each input channel.
        """

        super(SpatialMeans, self).__init__(**kwargs)

        # initialisation
        self.size_in = input_shape  # this doesn't include the batch size
        self.return_power = return_power

        # get input shape
        self.shape_non_chan = input_shape[1:]
        self.n_chan = input_shape[0]

        # build meshgrid of size [1, 1, 3, H*W*D]
        self.coord_idx_list = torch.meshgrid(*[torch.arange(0, ss) for ss in self.shape_non_chan])
        self.coord_idx_list = [torch.reshape(ten, [-1]) for ten in self.coord_idx_list]
        self.coord_idxs = add_axis(torch.stack(self.coord_idx_list), [0, 0])

    def forward(self, x):

        # flatten the input tensor, along image dimensions (so in the end we have [Batch, Channels, H*W*D])
        x = torch.reshape(x, [-1, self.n_chan, np.prod(self.shape_non_chan)])
        x = torch.abs(x)

        # get mean coordinates, weighted by the feature values
        numerator = torch.sum(add_axis(x, 2) * self.coord_idxs, dim=3)
        denominator = torch.sum(x.detach(), dim=2, keepdim=True)   # don't normalise gradients
        means_by_chan = numerator / (denominator + 1e-8)  # [B, K, 3]

        if self.return_power:
            power_by_chan = x.sum(dim=2, keepdim=True)  # [B, K, 1]
            return means_by_chan, power_by_chan
        else:
            return means_by_chan

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.coord_idxs = self.coord_idxs.to(*args, **kwargs)
        for idx in range(len(self.coord_idx_list)):
            self.coord_idx_list[idx].to(*args, **kwargs)
        return self
