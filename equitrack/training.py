import os
import re
import numpy as np
import torch.utils.data
from shutil import copy2

from equitrack import networks
from equitrack import loaders
from equitrack.losses import xfm_loss, image_loss, dice
from equitrack.utils import aff_to_field, interpolate, build_subject_dict

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def training(training_im_dir,
             val_im_dir,
             results_dir,
             net_type,
             image_size,
             training_lab_dir=None,
             val_lab_dir=None,
             rotation_range=90,
             shift_range=20,
             mean_05=False,
             min_perc=0.01,
             max_perc=99.99,
             max_noise_std=0.05,
             max_bias_std=0.3,
             bias_scale=0.06,
             gamma_std=0.15,
             n_channels=64,
             n_levels=5,
             n_conv=1,
             n_feat=64,
             feat_mult=1,
             kernel_size=5,
             last_activation='relu',
             closed_form_algo='numerical',
             batch_size=4,
             learning_rate=1e-5,
             n_epochs=100000,
             validate_every_n_epoch=100,
             weight_image_loss=1,
             weight_xfm_loss=0,
             xfm_loss_name='l2',
             weight_dice_loss=0,
             resume=False):
    """
    This function trains a network to rigidly register 2 input images (moving and fixed). For this, it uses a network to
    extract in parallel features from the two images, and estimate a rigid transform.
    There are several options for both parts of this framework, which are specified in the 'net_type' parameter.
        The feature extractor can either be:
                - an equivariant network (use 'se3' in name). Note that the architecture of this network is fixed,
                    except for the number of output channels.
                - or a conv net, with options: - a UNet ('conv_unet').
                                               - an encoder ('conv_encoder')
        The rigid registration can either:
                - collapse each feature map onto its center of mass, which gives us two point clouds. These are then
                    registered with a svd-based closed-form algorithm. Use 'svd' in the name.
                - a direct regression based on densely connected conv layers ('dense'). This is incompatible with 'se3',
                    and 'unet', as the dimension of the first densely connected layer would explode.
    This framework is trained on pairs simulated by spatially augmenting twice the same image, which gives us access to
    th exact ground truth transform. Both images can further be augmented separately with intensity corruptions.
    The training loss can be composed up to three parts: L2 image unsupervised loss, supervised transformation loss, and
    supervised Dice loss on provided masks.
    :param training_im_dir: path of the directory with training images
    :param val_im_dir: path of the directory with validation images
    :param results_dir: path of the directory where all training models, losses, and intermediate results will be saved.
    :param net_type: name of the architecture. Can be 'se3_svd', 'conv_unet_svd', or 'conv_encoder_dense'.
    :param image_size: size of the training input data. If some images in training/validation dir are not of this size,
    they will be zero-padded and cropped to fit image_size. This can be an integer (same size in all three directions),
    or a sequence of length 3.
    :param training_lab_dir: (optional) path of the directory with training masks. These are used to regularise training
    by computing the Dice loss between fixed and moved masks.
    :param val_lab_dir: (optional) path of the directory with validation masks. These are used to compute Dice as a
    complementary validation metric.
    :param rotation_range: (optional) range for the rotation augmentation. Must be a float/integer where angles will be
    sampled uniformly in [-rotation_range, +rotation_range]. Defaults to [-90,90].
    :param shift_range: (optional) same as above but for translations. Defaults to [-20,20].
    :param mean_05: (optional) whether to further re-center the average intensity to 0.5 after normalisation
    (default is False).
    :param min_perc: (optional) During training, image intensities are normalised in [0,1] with min-max normalisation.
    min_perc is the minimum percentile that will correspond to 0 after normalisation. Defaults to 0.01.
    :param max_perc: (optional) same as above but for maximum value. Defaults to 99.99.
    :param max_noise_std: (optional) maximum standard deviation of the Gaussian noise to apply (higher = stronger).
    Default is 0.05.
    :param max_bias_std: (optional) maximum standard deviation of the bias field corruption to apply
    (higher = stronger). Default is 0.3.
    :param bias_scale: (optional) scale of the bias field (lower = smoother). Default is 0.06.
    :param gamma_std: (optional) maximum value of the gamma-exponentiation for histogram shifting (higher = stronger).
    Default is 0.15.
    :param n_channels: (optional) number of output channels of the feature extractor. Default is 32.
    :param n_levels: (optional) number of resolution levels (only used if 'conv' is in net_type).
    :param n_conv: (optional) number of convolution per resolution level (only used if 'conv' is in net_type).
    :param n_feat: (optional) number of initial feature maps after the first convolution (only used if 'conv' is in
    net_type).
    :param feat_mult: (optional) feature multiplier after each max pooling (only used if 'conv' is in net_type).
    :param kernel_size: (optional) size of convolutional kernels  (only used if 'conv' is in net_type).
    :param last_activation: (optional) last non-linearity (only used if 'conv' is in net_type).
    :param closed_form_algo: (optional) type of the closed-form algorithm to use. Can be 'numerical' (default)
    or 'analytical', which implements the strategy used in KeyMorph.
    :param batch_size: (optional) number of examples per training batch. Default is 4, but change this to fit GPU memory
    :param learning_rate: (optional) learning rate for the Adam optimiser. Default is 10^-5.
    :param n_epochs: (optional) number of training epochs (set this to a high value to ensure convergence). Default is
    100000, which is waaaayyyy beyond convergence (I always stop training before this).
    :param validate_every_n_epoch: (optional) frequency of the validation step for model evaluation. Default is 100.
    :param weight_image_loss: (optional) weight of the unsupervised L2 image loss. Default is 1.
    :param weight_xfm_loss: (optional) weight of the supervised transform loss. Not used by default.
    :param xfm_loss_name: (optional) type of the transform loss to apply. Can either be 'l2' or 'geodesic'. Not used
    by default.
    :param weight_dice_loss: (optional) weight of the supervised Dice loss. If >0, training_lab_dir must be given.
    Not used by default.
    :param resume: (optional) whether to resume training where we left off. Training will resume from the last model
    written in results_dir. Not used by default.
    """

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size

    # create result directory
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # set up augmenter params
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size
    augment_params = {'resize': image_size,
                      'rotation_range': rotation_range,
                      'shift_range': shift_range,
                      'crop_size': None,
                      'flip': False,
                      'min_perc': min_perc,
                      'max_perc': max_perc,
                      'mean_05': mean_05,
                      'max_noise_std': max_noise_std,
                      'max_bias_std': max_bias_std,
                      "bias_scale": bias_scale,
                      "gamma_std": gamma_std}

    # training loader
    print('create training loader...', flush=True)
    training_subj_dict = build_subject_dict(training_im_dir, training_lab_dir)
    train_dataset = loaders.loader_rxfm(subj_dict=training_subj_dict,
                                        augm_params=augment_params,
                                        return_masks=(training_lab_dir is not None),
                                        return_clean_images=(weight_image_loss > 0),
                                        seed=1919)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # validation loader
    print('create validation loader...', flush=True)
    val_subj_dict = build_subject_dict(val_im_dir, val_lab_dir)
    val_dataset = loaders.loader_rxfm(subj_dict=val_subj_dict,
                                      augm_params=augment_params,
                                      validator_mode=True,
                                      return_masks=(val_lab_dir is not None),
                                      seed=1919)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    # initialise feature extractor
    print('initialise architecture...', flush=True)
    net = networks.Archi(name=net_type,
                         input_shape=image_size,
                         n_out_chan=n_channels,
                         n_levels=n_levels,
                         n_conv=n_conv,
                         n_feat=n_feat,
                         feat_mult=feat_mult,
                         kernel_size=kernel_size,
                         last_activation=last_activation,
                         closed_form_algo=closed_form_algo)
    net = net.to(device)

    # initialise optimizer
    print('initialise optimizer...\n', flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Check whether to resume training or load weights pretraining
    last_epoch = 0
    best_val_loss = 1e9
    best_val_dice = 0
    list_scores = []
    if resume:
        previous_files = sorted([p for p in os.listdir(models_dir) if re.sub('\D', '', p) != ''])
        if len(previous_files) > 0:
            print(f'loading from {previous_files[-1]}', flush=True)
            checkpoint = torch.load(os.path.join(models_dir, previous_files[-1]), map_location=torch.device(device))
            net.load_state_dict(checkpoint['net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            path_val_scores = os.path.join(results_dir, 'val_scores.npy')
            if os.path.isfile(path_val_scores):
                list_scores = np.load(path_val_scores)
                best_val_loss = np.min(list_scores[:, 2])
                best_val_dice = np.max(list_scores[:, 3])
                list_scores = list_scores.tolist()

    # Training loop
    for epoch in range(last_epoch, n_epochs):
        print('Epoch', epoch, flush=True)

        net.train()
        epoch_train_loss = 0
        for i, batch in enumerate(train_loader):

            # initialise inputs
            clean_moving, clean_fixed, mask_moving, mask_fixed = (None, None, None, None)
            moving = batch['scan_moving'].to(device)
            fixed = batch['scan_fixed'].to(device)
            xfm_gt = batch['xfm'].to(device)
            if weight_image_loss > 0:
                clean_moving = batch['clean_scan_moving'].to(device)
                clean_fixed = batch['clean_scan_fixed'].to(device)
            if weight_dice_loss > 0:
                mask_moving = batch['mask_moving'].to(device)
                mask_fixed = batch['mask_fixed'].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                # get predictions
                xfm = net.forward((moving, fixed))

                # compute loss
                grid_xfm, clean_moved, mask_moved = (None, None, None)
                if (weight_xfm_loss <= 0) & (weight_image_loss <= 0) & (weight_dice_loss <= 0):
                    raise ValueError('at least one of weight_xfm_loss, weight_image_loss, or weight_dice_loss must > 0')
                if weight_xfm_loss > 0:
                    train_loss = xfm_loss(xfm_gt, xfm, loss_type=xfm_loss_name)
                else:
                    train_loss = 0
                if (weight_image_loss > 0) | (weight_dice_loss > 0):
                    grid_xfm = aff_to_field(xfm, image_size, invert_affine=True)
                if weight_image_loss > 0:
                    clean_moved = interpolate(torch.moveaxis(clean_moving, 1, -1), grid_xfm, batch_size)
                    clean_moved = torch.moveaxis(clean_moved, -1, 1)
                    train_loss = train_loss + weight_image_loss * image_loss(clean_fixed, clean_moved)
                if weight_dice_loss > 0:
                    mask_moved = interpolate(torch.moveaxis(mask_moving, 1, -1), grid_xfm, batch_size, 'nearest')
                    mask_moved = torch.moveaxis(mask_moved, -1, 1)
                    train_loss = train_loss + weight_dice_loss * dice(mask_fixed, mask_moved)

                # backpropagation
                train_loss.backward(retain_graph=True)
                optimizer.step()

            # print iteration info
            epoch_train_loss += train_loss.item() * batch_size
            print('iteration:{}/{}  loss:{:.5f}'.format(i + 1, len(train_dataset), train_loss.item()), flush=True)

            # flush cuda memory
            del moving, fixed, xfm_gt, clean_moving, clean_fixed, mask_moving, mask_fixed, xfm, \
                train_loss, grid_xfm, clean_moved, mask_moved
            torch.cuda.empty_cache()

        epoch_train_loss = epoch_train_loss / len(train_dataset)

        # save and validate model every 5 epochs, otherwise just print training info
        if epoch % validate_every_n_epoch != 0:
            print('Epoch:{}  Train Loss:{:.5f}'.format(epoch, epoch_train_loss) + '\n', flush=True)

        else:
            torch.save({'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch},
                       os.path.join(models_dir, '%05d.pth' % epoch))

            # eval loop
            net.eval()
            epoch_val_loss = 0
            epoch_val_dice = 0
            for j, batch in enumerate(val_loader):

                # initialise inputs
                mask_moving, mask_fixed = (None, None,)
                moving = batch['scan_moving'].to(device)
                fixed = batch['scan_fixed'].to(device)
                if val_lab_dir is not None:
                    mask_moving = batch['mask_moving'].to(device)
                    mask_fixed = batch['mask_fixed'].to(device)
                xfm_gt = batch['xfm'].to(device)

                # predict transformation
                xfm = net.forward((moving, fixed))

                # compute transformation loss
                val_loss = xfm_loss(xfm_gt, xfm, loss_type=xfm_loss_name)
                epoch_val_loss += val_loss.item()

                # compute dice coef
                if mask_moving is not None:
                    grid_xfm = aff_to_field(xfm, image_size, invert_affine=True)
                    mask_moved = interpolate(torch.moveaxis(mask_moving, 1, -1), grid_xfm, 1, 'nearest', torch.int32)
                    mask_moved = torch.moveaxis(mask_moved, -1, 1)
                    epoch_val_dice += dice(mask_fixed, mask_moved).mean().item()
                    del grid_xfm, mask_moved

                # flush cuda memory
                del moving, fixed, mask_moving, mask_fixed, xfm_gt, xfm, val_loss
                torch.cuda.empty_cache()

            # save validation scores
            if val_lab_dir is not None:
                epoch_val_loss = epoch_val_loss / len(val_dataset)
                epoch_val_dice = epoch_val_dice / len(val_dataset)
                print('Epoch:{}  Train Loss:{:.5f}  Val Loss:{:.5f}  Dice:{:.3f}'.format(
                    epoch, epoch_train_loss, epoch_val_loss, epoch_val_dice) + '\n', flush=True)
                list_scores.append([epoch, epoch_train_loss, epoch_val_loss, epoch_val_dice])
                np.save(os.path.join(results_dir, 'val_scores.npy'), np.array(list_scores))
            else:
                epoch_val_loss = epoch_val_loss / len(val_dataset)
                print('Epoch:{}  Train Loss:{:.5f}  Val Loss:{:.5f}'.format(
                    epoch, epoch_train_loss, epoch_val_loss) + '\n', flush=True)
                list_scores.append([epoch, epoch_train_loss, epoch_val_loss])
                np.save(os.path.join(results_dir, 'val_scores.npy'), np.array(list_scores))

            # save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                copy2(os.path.join(models_dir, '%05d.pth' % epoch), os.path.join(results_dir, 'best_val_loss.pth'))
                with open(os.path.join(results_dir, 'best_epoch_val_loss.txt'), 'w') as f:
                    f.write('epoch:%d   val loss:%f' % (epoch, best_val_loss))
            if epoch_val_dice > best_val_dice:
                best_val_dice = epoch_val_dice
                copy2(os.path.join(models_dir, '%05d.pth' % epoch), os.path.join(results_dir, 'best_dice.pth'))
                with open(os.path.join(results_dir, 'best_epoch_dice.txt'), 'w') as f:
                    f.write('epoch:%d   dice coef:%f' % (epoch, best_val_dice))

    del net
