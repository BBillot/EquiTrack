from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np

from equitrack.utils import apply_intensity_transform, create_transform, aff_to_field, interpolate


class SpatialAugmenter(object):
    """Spatial augmenter for training and validation inputs.
    In training, all augmentation parameters are randomly sampled at each mini-batch. For validation,
    these parameters are subject-specific but fixed across time, and thus given as inputs in list_of_xfm_params."""

    def __init__(self,
                 list_of_xfm_params=None,
                 rotation_range=0.,
                 shift_range=0.,
                 crop_size=None,
                 flip=False,
                 seed=None,
                 return_affine=False,
                 track_flip_number=False,
                 use_max_values=False,
                 normalise=False,
                 mean_05=True):
        """
        :param list_of_xfm_params: list of dict with {"rotation": r, "translation": t}, where r and t are
        length-3 numpy arrays. (used for validation)
        :param rotation_range: maximum rotation angle for augmentation (in degrees)
        :param shift_range: maximum shift for augmentation (in voxels)
        :param crop_size: randomly crop examples to size (given as list [H, W, D]), None=no cropping
        :param flip: randomly flip examples
        :param seed: numpy seed to keep the augmentation the same across training runs.
        :param return_affine: whether to return the applied rigid transform in homogeneous representation
        :param track_flip_number: whether to track the number of flips
        :param use_max_values: whether to use the maximum rotation and translation values instead on drawing them from
        uniform distributions
        :param normalise: whether to rescale the inputs in [0,1]
        :param mean_05: recenter intensities after rescaling to have 0.5 mean
        """

        # initialisation
        self.list_of_xfm_params = list_of_xfm_params  # validation case
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.crop_size = crop_size
        self.flip = flip

        self.return_affine = return_affine
        self.track_flip_number = track_flip_number
        self.use_max_values = use_max_values

        self.normalise = normalise
        self.mean_05 = mean_05

        # enforce numpy seed
        if seed is not None:
            np.random.seed(seed)

    def random_transform(self, *args):
        """Randomly rotate/translate/crop/flip an image tensor of shape [H, W, D, C], and optionally its labels."""

        # get transformation matrix
        if self.rotation_range:
            if self.use_max_values:
                rx, ry, rz = self.rotation_range, self.rotation_range, self.rotation_range
            else:
                rx, ry, rz = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        else:
            rx, ry, rz = 0, 0, 0
        if self.shift_range:
            if self.use_max_values:
                tx, ty, tz = self.shift_range, self.shift_range, self.shift_range
            else:
                tx, ty, tz = np.random.uniform(-self.shift_range, self.shift_range, 3)
        else:
            tx, ty, tz = 0, 0, 0
        transform_matrix = create_transform(rx, ry, rz, tx, ty, tz, ordering='txyz')

        return self.perform_transform(transform_matrix, *args)

    def predefined_transform(self, idx, *args):
        """Rotate/translate/crop/flip an image tensor of shape [H, W, D, C], and optionally its labels,
        with parameters given beforehand. Used for validation."""

        # get rotation and translation params
        rotation = self.list_of_xfm_params[idx]["rotation"]
        translation = self.list_of_xfm_params[idx]["translation"]
        transform_matrix = create_transform(*rotation, *translation, ordering='txyz')

        # apply augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
        return self.perform_transform(transform_matrix, *args)

    def perform_transform(self, transform_matrix, *args):

        # apply transform, computation is done with torch but inputs and outputs are numpy
        with torch.no_grad():
            outputs = []
            for vol_idx, x in enumerate(args):
                if vol_idx > 0:
                    method = "nearest"
                    dtype = torch.int32
                else:
                    method = "linear"
                    dtype = torch.float32
                grid = aff_to_field(transform_matrix, x.shape[:3], invert_affine=True)
                x = interpolate(x, grid, method=method, vol_dtype=dtype)
                outputs.append(x)

            if self.normalise:
                outputs[0] = torch.clamp(outputs[0], 0)
                m = torch.min(outputs[0])
                M = torch.max(outputs[0])
                outputs[0] = (outputs[0] - m) / (M - m + 1e-9)
                if self.mean_05:
                    outputs[0] = outputs[0] / torch.mean(outputs[0][outputs[0] > 0]) * 0.5

            outputs = [out.detach().numpy() for out in outputs]

        # cropping
        if self.crop_size:
            cx = np.random.randint(0, args[0].shape[0] - self.crop_size[0])
            cy = np.random.randint(0, args[0].shape[1] - self.crop_size[1])
            cz = np.random.randint(0, args[0].shape[2] - self.crop_size[2])
            for vol_idx in range(len(args)):
                outputs[vol_idx] = outputs[vol_idx][cx:cx + self.crop_size[0],
                                                    cy:cy + self.crop_size[1],
                                                    cz:cz + self.crop_size[2], :]

        # flipping
        flip_number = 0
        if self.flip:
            for axis in range(3):
                if np.random.random() < 0.5:
                    flip_number += 1
                    for vol_idx in range(len(args)):
                        outputs[vol_idx] = np.flip(outputs[vol_idx], axis)

        # return outputs
        outputs = [outputs]
        if self.return_affine:
            outputs.append(transform_matrix)
        elif self.track_flip_number:
            outputs.append(flip_number)
        return outputs[0] if len(outputs) == 1 else outputs  # [H, W, D, C]


class IntensityAugmenter(object):
    """Intensity augmenter for training and validation inputs.
    In training, all augmentation parameters are randomly sampled at each mini-batch. For validation,
    these parameters are subject-specific but fixed across time, and thus given as inputs in list_of_xfm_params."""

    def __init__(self,
                 list_of_params=None,
                 max_noise_std=0.,
                 max_bias_std=0.,
                 bias_scale=0.06,
                 gamma_std=0.,
                 use_max_values=False,
                 mean_05=False,
                 seed=None):
        """
        :param list_of_params: list of dict with {"noise_field": noise, "bias_field": bias, "gamma": g},
        where noise is a field of the same shape as x that will be added to it (additive noise)
              bias is a small field, that will be resampled to image size and multiplied to x
              g is a scalar by which all voxels of x will be exponentiated
        :param max_noise_std: maximum standard deviation for the Gaussian noise
        :param max_bias_std: maximum std. dev for the bias filed corruption (higher = strobger corruption)
        :param bias_scale: scale of the bias field (lower = smoother)
        :param gamma_std: std dev for random exponentiation (higher = stronger)
        :param use_max_values: whether to use the maximum rotation and translation values instead on drawing them from
        uniform distributions
        :param mean_05: recenter intensities after rescaling to have 0.5 mean
        :param seed: numpy seed to keep the augmentation the same across training runs.
        """

        # initialise
        self.list_of_intensity_params = list_of_params
        self.max_noise_std = max_noise_std
        self.max_bias_std = max_bias_std
        self.bias_scale = bias_scale
        self.gamma_std = gamma_std
        self.use_max_values = use_max_values
        self.mean_05 = mean_05

        # enforce numpy seed
        if seed is not None:
            np.random.seed(seed)

    def random_transform(self, x):
        """Randomly corrupt an image tensor of shape [H, W, D, C] with noise/bias """

        if self.max_noise_std > 0 or self.max_bias_std > 0 or self.gamma_std > 0:

            # sample noise field
            if self.use_max_values:
                noise_std = self.max_noise_std
            else:
                noise_std = np.random.uniform(high=self.max_noise_std)
            if noise_std > 0:
                noise_field = np.random.normal(0, noise_std, x.shape[:3])
            else:
                noise_field = None

            # sample small bias field
            if self.use_max_values:
                bias_std = self.max_bias_std
            else:
                bias_std = np.random.uniform(high=self.max_bias_std)
            if bias_std > 0:
                bias_sample_size = [math.ceil(size * self.bias_scale) for size in x.shape[:3]]
                bias_field = np.random.normal(0, bias_std, bias_sample_size)
            else:
                bias_field = None

            # sample gamma
            if self.gamma_std > 0:
                if self.use_max_values:
                    gamma = self.gamma_std
                else:
                    gamma = np.random.normal(scale=self.gamma_std)
            else:
                gamma = None

            # apply intensity augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
            x = apply_intensity_transform(x, noise_field, bias_field, gamma, self.mean_05)

        return x

    def predefined_transform(self, idx, x):
        """Corrupt an image tensor of shape [H, W, D, C] with noise/bias fields computed beforehand.
        Used for validation"""

        # get noise and bias fields
        noise_field = self.list_of_intensity_params[idx]["noise_field"]
        bias_field = self.list_of_intensity_params[idx]["bias_field"]
        gamma = self.list_of_intensity_params[idx]["gamma"]

        # apply intensity augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
        if noise_field is not None or bias_field is not None or gamma is not None:
            x = apply_intensity_transform(x, noise_field, bias_field, gamma, self.mean_05)

        return x
