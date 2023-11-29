import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.image.gated_block import GatedBlock

from energi import spatial_mean as sm
from energi.utils import pts_to_xfm_numerical, pts_to_xfm_analytical, create_transform


class Archi(nn.Module):

    def __init__(self,
                 name,
                 input_shape,
                 n_out_chan,
                 n_levels=5,
                 n_conv=4,
                 n_feat=64,
                 feat_mult=1,
                 kernel_size=3,
                 last_activation=None,
                 return_inputs=False,
                 closed_form_algo='numerical'):
        """
        This class builds the feature extractor network and the algorithm to estimate rigid transforms.
        There are several options for both parts of the network, which are passed here in the name parameter.
        1 can either be:
                - an equivariant network (put 'se3' in name),
                - or a conv net, with options: - a UNet ('conv_unet').
                                               - an encoder ('conv_encoder')
        2 can either be:
                - a svd-based closed-form solution to find the optimal transform between two clouds of points, which
                    are here the center of masses of each feature map ('svd'). Actually here we can either use an 'svd'
                    based algorithm ('numerical'), or a different algo derived in KeyMorph 'analytical'.
                - a direct regression based on densely connected conv layers ('dense'). This is incompatible with 'se3',
                    and 'unet', as the dimension of the first densely connected layer would explode.
        Finally, 1 can either process the two images to register in parallel, or concatenate them and process them
        together ('concat'), although this is not compatible with 'se3' or 'svd'.

        Note that you can only play with the architecture of the conv networks, since se3 is fixed.

        :param name: name of the architecture. Can be 'se3_svd', 'conv_unet_svd', 'conv_encoder_dense', or
        'conv_encoder_dense_concat'
        :param input_shape: shape of inputs, should be a list [H, W, D]
        :param n_out_chan: number of output channels for the feature extractor
        :param n_levels: number of resolution levels for the feature extractor 'conv' nets
        :param n_conv: number of convolution per resolution level for the feature extractor in 'conv' nets
        :param n_feat: number of initial feature maps after the first convolution for the feat. extractor in 'conv' nets
        :param feat_mult: feature multiplier after each max pooling for the feature extractor in 'conv' nets
        :param kernel_size: size of convolutional kernels for the feature extractor in 'conv' nets
        :param last_activation: last activation of the feature extractor in 'conv' nets. default is None.
        :param return_inputs: whether to return the inputs of the network as additional outputs
        (can be useful when this model is appended to a denoiser, so that we can see the output of the denoiser)
        :param closed_form_algo: can either be 'numerical' or 'analytical'
        """

        super(Archi, self).__init__()

        if ('concat' in name) & (('se3' in name) | ('svd' in name)):
            raise ValueError('concat cannot be used with se3 or svd')
        if ('dense' in name) & (('se3' in name) | ('unet' in name)):
            raise ValueError('se3, unet, and flat cannot be used with dense, otherwise the first dense layer explodes')

        self.name = name
        self.img_shape_tensor = torch.tensor(input_shape)  # [H, W, D]
        self.return_inputs = return_inputs
        self.closed_form_algo = closed_form_algo
        n_out_chan = None if n_out_chan == 0 else n_out_chan

        # feature extraction
        if 'se3' in self.name:  # rotation-equivariant net
            self.main_net = RXFM_Net(n_out_chan)
        elif 'conv' in self.name:  # normal conv
            n_in_chan = 2 if 'concat' in self.name else 1  # concatenated inputs processed together
            if 'unet' in self.name:
                self.main_net = UNet(n_input_channels=n_in_chan,
                                     n_output_channels=n_out_chan,
                                     n_levels=n_levels,
                                     n_conv=n_conv,
                                     n_feat=n_feat,
                                     feat_mult=feat_mult,
                                     kernel_size=kernel_size,
                                     last_activation=last_activation,
                                     batch_norm_after_each_conv='flat' in self.name)
            elif 'encoder' in self.name:
                if 'svd' in self.name:
                    upsample = True
                else:
                    upsample = False
                    n_out_chan = n_feat * feat_mult ** (n_levels - 1)
                self.main_net = UNet(encoder_only=True,
                                     n_input_channels=n_in_chan,
                                     n_output_channels=n_out_chan,
                                     n_levels=n_levels,
                                     n_conv=n_conv,
                                     n_feat=n_feat,
                                     feat_mult=feat_mult,
                                     kernel_size=kernel_size,
                                     last_activation=last_activation,
                                     upsample=upsample)
            else:
                raise ValueError('if conv in name, architecture should be given: unet, encoder, flat')
        else:
            raise ValueError('network should either be se3 or conv, name was %s' % self.name)

        # transformation parameter regression
        if 'svd' in self.name:  # closed-form solution
            self.end_net = sm.SpatialMeans([n_out_chan] + input_shape,
                                           return_power=self.closed_form_algo == 'numerical')
        elif 'dense' in self.name:  # fully connected layers
            fc_layers = list()
            list_feat = [int((np.prod(input_shape) / (8 ** (n_levels - 1))) * n_out_chan), 512, 512, 6]
            if 'concat' not in self.name:
                list_feat[0] *= 2
            for i in range(len(list_feat) - 1):
                fc_layers.append(torch.nn.Linear(list_feat[i], list_feat[i + 1]))
                if i < len(list_feat) - 2:
                    fc_layers.append(torch.nn.ReLU())
            self.end_net = torch.nn.Sequential(*fc_layers)
        else:
            raise ValueError('regression should either be svd or dense, name was %s' % self.name)

    def forward(self, x):

        # feature extraction
        moving, fixed = x
        if 'concat' in self.name:  # concatenated inputs processed together
            outputs_concat = self.main_net.forward(torch.cat([moving, fixed], 1))  # [B, K, H, W, D]
            features_moving = features_fixed = None
        else:  # siamese networks
            features_moving = self.main_net.forward(moving)  # [B, K, H, W, D]
            features_fixed = self.main_net.forward(fixed)
            outputs_concat = torch.cat((features_moving, features_fixed), 1)

        # rigid transform regression
        if 'svd' in self.name:  # closed-form solution
            if self.closed_form_algo == 'numerical':
                means_moving, weights_moving = self.end_net.forward(features_moving)  # mean [B, K, 3] weights [B, K, 1]
                means_fixed, weights_fixed = self.end_net.forward(features_fixed)
                weights = weights_moving * weights_fixed
                weights = weights / (weights.sum(dim=1, keepdims=True) + 1e-8)
                try:
                    xfm = pts_to_xfm_numerical(means_moving, means_fixed, weights, self.img_shape_tensor)  # [B, 4, 4]
                except RuntimeError:
                    xfm = pts_to_xfm_analytical(means_moving, means_fixed, self.img_shape_tensor)  # [B, 4, 4]
            else:
                means_moving = self.end_net.forward(features_moving)  # [B, K, 3]
                means_fixed = self.end_net.forward(features_fixed)
                xfm = pts_to_xfm_analytical(means_moving, means_fixed, self.img_shape_tensor)  # [B, 4, 4]
        else:  # fully connected layers
            xfm = self.end_net(torch.flatten(outputs_concat, start_dim=1))  # [B, 6]
            angles = torch.split(torch.tanh(xfm[..., 0:3]) * np.pi, 1, 1)  # list of 3 [B, 1]
            shifts = torch.split(xfm[..., 3:], 1, 1)  # list of 3 [B, 1]
            xfm = create_transform(*angles, *shifts, input_angle_unit='rad')  # [B, 4, 4]

        outputs = [xfm]
        if self.return_inputs:
            outputs += [moving, fixed]
        return outputs if len(outputs) > 1 else outputs[0]

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.main_net = self.main_net.to(*args, **kwargs)
        self.end_net = self.end_net.to(*args, **kwargs)
        self.img_shape_tensor = self.img_shape_tensor.to(*args, **kwargs)
        return self


class DenoiserWrapper(nn.Module):
    """Wrapper around a UNet, which will process 2 inputs in parallel, and renormalise the intensities of the outputs"""

    def __init__(self,
                 n_levels,
                 n_conv=4,
                 n_feat=32,
                 feat_mult=1,
                 kernel_size=3,
                 rm_top_skip_connection=0,
                 predict_residual=True,
                 postprocess_outputs=True):
        """
        :param n_levels: number of resolution levels
        :param n_conv: number of convolution per resolution level
        :param n_feat: number of initial feature maps after the first convolution
        :param feat_mult: feature multiplier after each max pooling
        :param kernel_size: size of convolutional kernels
        :param rm_top_skip_connection: whether to remove the top skip connection. Default is 0 where none are removed
        :param predict_residual: whether to add a residual connection between the input and the last layer
        :param postprocess_outputs: whether to rescale the intensities of the denoised volumes between [0,1]
        """

        super(DenoiserWrapper, self).__init__()
        self.denoiser = UNet(n_levels=n_levels,
                             n_conv=n_conv,
                             n_feat=n_feat,
                             feat_mult=feat_mult,
                             kernel_size=kernel_size,
                             rm_top_skip_connection=rm_top_skip_connection,
                             predict_residual=predict_residual)
        self.postprocess_outputs = postprocess_outputs

    def forward(self, x):
        """takes tuple of 2 inputs, each with the same shape [B, C, H, W, D]"""
        moving, fixed = x

        # denoise image
        moving_denoised = self.denoiser.forward(moving)
        fixed_denoised = self.denoiser.forward(fixed)

        # mask out the rest of the image and rescale between 0 and 1
        if self.postprocess_outputs:
            moving_denoised = torch.clamp(moving_denoised, min=0)
            fixed_denoised = torch.clamp(fixed_denoised, min=0)

            moving_denoised = moving_denoised * (moving > 0).to(dtype=moving_denoised.dtype)
            fixed_denoised = fixed_denoised * (fixed > 0).to(dtype=fixed_denoised.dtype)

            min_moving = torch.amin(moving_denoised, dim=[1, 2, 3, 4], keepdim=True)
            max_moving = torch.amax(moving_denoised, dim=[1, 2, 3, 4], keepdim=True)
            moving_denoised = (moving_denoised - min_moving) / (max_moving - min_moving)

            min_fixed = torch.amin(fixed_denoised, dim=[1, 2, 3, 4], keepdim=True)
            max_fixed = torch.amax(fixed_denoised, dim=[1, 2, 3, 4], keepdim=True)
            fixed_denoised = (fixed_denoised - min_fixed) / (max_fixed - min_fixed)

        return moving_denoised, fixed_denoised

    def get_channel_outputs(self, x):
        return self.denoiser.forward(x)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.denoiser = self.denoiser.to(*args, **kwargs)
        return self


class RXFM_Net(nn.Module):
    """Equivariant network. It takes as input an image of shape [B, C, H, W, D] and returns equivariant feature maps
    of size [B, output_chans, H, W, D]. The network has a fixed architecture where the only tunable parameter is the
    number of output features output_chans."""

    def __init__(self, output_chans):
        super(RXFM_Net, self).__init__()

        # number of features per layer
        features = [[1]] + [[16, 16, 4]] * 4 + [[output_chans]]

        # convolution parameters
        common_block_params = {'size': 5,
                               'padding': 2,  # = 5 // 2
                               'normalization': None,
                               'capsule_dropout_p': None,
                               'smooth_stride': False,
                               'activation': F.relu}

        # build sequential model of convolutions
        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params) for i in range(len(features) - 1)]
        self.sequence = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.sequence(x.to(dtype=torch.float32))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self


class UNet(nn.Module):
    """UNet architecture"""

    def __init__(self,
                 n_input_channels=1,
                 n_output_channels=1,
                 n_levels=3,
                 n_conv=2,
                 n_feat=32,
                 feat_mult=1,
                 kernel_size=3,
                 activation='relu',
                 last_activation=None,
                 batch_norm_after_each_conv=False,
                 residual_blocks=False,
                 encoder_only=False,
                 upsample=False,
                 use_skip_connections=True,
                 rm_top_skip_connection=0,
                 predict_residual=False):
        """
        :param n_input_channels: number of input channels
        :param n_output_channels: number of output channels (i.e. feature maps)
        :param n_levels: number of resolution levels
        :param n_conv: number of convolution per resolution level
        :param n_feat: number of initial feature maps after the first convolution
        :param feat_mult: feature multiplier after each max pooling
        :param kernel_size: size of convolutional kernels
        :param activation: non-linearity to use. Can be 'relu' or 'elu'
        :param last_activation: last non-linearity before softmax. Can be 'relu' or 'elu', or None
        :param batch_norm_after_each_conv: if false, batch norm will be performed at teh end of each resolution level
        :param residual_blocks: whether to use residual connection at the end of each block
        :param encoder_only: do not add a decoder
        :param upsample: if encoder only, whether to upsample the bottleneck to the size of the inputs
        :param use_skip_connections: whether to use skip connections at all
        :param rm_top_skip_connection: whether to remove the top skip connection. Default is 0 where none are removed
        :param predict_residual: whether to add a residual connection between the input and the last layer
        """

        super(UNet, self).__init__()

        # input/output channels
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        # general architecture
        self.encoder_only = encoder_only
        self.upsample = upsample
        self.rm_top_skip_connection = rm_top_skip_connection if use_skip_connections else self.n_levels
        self.predict_residual = predict_residual

        # convolution block parameters
        self.n_levels = n_levels
        self.n_conv = n_conv
        self.feat_mult = feat_mult
        self.feat_list = [n_feat * feat_mult ** i for i in range(self.n_levels)]
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm_after_each_conv = batch_norm_after_each_conv
        self.residual_blocks = residual_blocks

        # define convolutional blocks
        self.list_encoder_blocks = self.get_list_encoder_blocks()  # list of length self.n_levels
        if not self.encoder_only:
            self.list_decoder_blocks = self.get_list_decoder_blocks()  # list of length self.n_levels - 1
            self.last_conv = torch.nn.Conv3d(self.feat_list[0], self.n_output_channels, kernel_size=1)
        else:
            self.list_decoder_blocks = []
            self.last_conv = torch.nn.Conv3d(self.feat_list[-1], self.n_output_channels, kernel_size=1)

        if last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif last_activation == 'elu':
            self.last_activation = torch.nn.ELU()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax()
        elif last_activation == 'tanh':
            self.last_activation = torch.nn.Tanh()
        else:
            self.last_activation = None

    def forward(self, x):
        """takes an input of shape [B, C, H, W, D]"""

        tens = x

        # down-arm
        list_encoders_features = []
        for i, encoder_block in enumerate(self.list_encoder_blocks):
            if i > 0:
                tens = torch.nn.functional.max_pool3d(tens, kernel_size=2)
            tens_out = encoder_block(tens)
            tens = tens + tens_out if self.residual_blocks else tens_out
            list_encoders_features.append(tens)

        # up-arm
        if not self.encoder_only:

            # remove output of last encoder block (i.e. the bottleneck) from the list of features to be concatenated
            list_encoders_features = list_encoders_features[::-1][1:]

            # build conv
            for i in range(len(self.list_decoder_blocks)):
                tens = torch.nn.functional.interpolate(tens, scale_factor=2, mode='trilinear')
                if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                    tens_out = torch.cat((list_encoders_features[i], tens), dim=1)
                else:
                    tens_out = tens
                tens_out = self.list_decoder_blocks[i](tens_out)
                tens = tens + tens_out if self.residual_blocks else tens_out

        # final convolution
        tens = self.last_conv(tens)
        if self.last_activation is not None:
            tens = self.last_activation(tens)

        if self.upsample:
            tens = torch.nn.functional.interpolate(tens, scale_factor=2 ** (self.n_levels - 1), mode='trilinear')

        # residual
        if self.predict_residual:
            tens = x + tens

        return tens

    def get_list_encoder_blocks(self):

        list_encoder_blocks = []
        for i in range(self.n_levels):

            # number of input/output feature maps for each convolution
            if i == 0:
                n_input_feat = [self.n_input_channels] + [self.feat_list[i]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[i - 1]] + [self.feat_list[i]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[i]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_encoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_encoder_blocks)

    def get_list_decoder_blocks(self):

        list_decoder_blocks = []
        for i in range(0, self.n_levels - 1):

            # number of input/output feature maps for each convolution
            if i < (self.n_levels - 1 - self.rm_top_skip_connection):
                n_input_feat = [self.feat_list[::-1][i + 1] * (1 + self.feat_mult)] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            else:
                n_input_feat = [self.feat_list[::-1][i]] + \
                               [self.feat_list[::-1][i + 1]] * (self.n_conv - 1)
            n_output_feat = self.feat_list[::-1][i + 1]

            # build conv block
            layers = self.build_block(n_input_feat, n_output_feat)
            list_decoder_blocks.append(torch.nn.Sequential(*layers))

        return nn.ModuleList(list_decoder_blocks)

    def build_block(self, n_input_feat, n_output_feat):

        # convolutions + activations
        layers = list()
        for conv in range(self.n_conv):
            layers.append(torch.nn.Conv3d(n_input_feat[conv], n_output_feat, kernel_size=self.kernel_size,
                                          padding=self.kernel_size // 2))
            if self.activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif self.activation == 'elu':
                layers.append(torch.nn.ELU())
            else:
                raise ValueError('activation should be relu or elu, had: %s' % self.activation)
            if self.batch_norm_after_each_conv:
                layers.append(torch.nn.BatchNorm3d(n_output_feat))

        # batch norm
        if not self.batch_norm_after_each_conv:
            layers.append(torch.nn.BatchNorm3d(n_output_feat))

        return layers

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        return self
