import os
import numpy as np
import torch.utils.data
from pytorch3d.transforms import matrix_to_euler_angles as matrix_to_angles

from equitrack import loaders
from equitrack import networks
import equitrack.losses as losses
from equitrack.utils import build_subject_dict_series, build_xfm_dict_series, aff_to_field, interpolate, save_volume

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings("ignore")


def predict(path_main_model,
            main_data_dir,
            name_image_dir,
            net_type,
            image_size,
            name_xfm_dir=None,
            name_labels_dir=None,
            min_perc=0.01,
            max_perc=99.99,
            mean_05=False,
            main_n_channels=64,
            main_n_levels=4,
            main_n_conv=2,
            main_n_feat=16,
            main_feat_mult=2,
            main_kernel_size=5,
            main_last_activation='relu',
            closed_form_algo='numerical',
            path_denoiser_model=None,
            denoiser_n_levels=4,
            denoiser_n_conv=2,
            denoiser_n_feat=32,
            denoiser_feat_mult=1,
            denoiser_kernel_size=3,
            denoiser_rm_top_skip_connection=1,
            denoiser_predict_residual=True,
            recompute=True):
    """
    This function performs rigid motion tracking in time-series relatively to the first time frame of the series.
    As before, it has two parts: a feature extractor (an equivariant network or a CNN) and a rigid transform estimator
    (SVD-based closed-form algorithm or fully connected layers). In addition, one can prepend a denoising CNN to remove
    anatomically irrelevant features.

    This function expects the input data to be organised as follows:
    main_data_dir/time_series_1/name_image_dir: image_time_frame_0.nii.gz
                                                image_time_frame_1.nii.gz
                                              ...
                                name_label_dir: label_time_frame_0.nii.gz
                                                label_time_frame_1.nii.gz
                                                ...
                                name_xfm_dir:   gt_4x4_transform_matrix_1_to_0.npy
                                              ...
                  time_series_2/name_image_dir: image_time_frame_0.nii.gz
                                                image_time_frame_1.nii.gz
                                                ...
                                name_label_dir: label_time_frame_0.nii.gz
                                                label_time_frame_1.nii.gz
                                                ...
                                name_xfm_dir:   gt_4x4_transform_matrix_1_to_0.npy
                                                ...
    Note that the name_label_dir and name_xfm_dir are optional, and are used to compute test scores if provided.

    Results for each time-series are written in their individual directory as follows
    time_series_1/equitrack/predicted_transforms/predicted_transforms_4x4_matrix/*.npy: - rigid transform in 4x4
                                                                                       homogeneous matrices between
                                                                                       each frame and the first one
                                             /predicted_rotation_angles.npy:         - summary of the predicted angles
                                                                                       for all time frames
                                             /predicted_translation_shifts.npy:      - summary of the predicted shifts
                                                                                       for all time frames
               /test_images/denoised,inputs,masks:                                   - here we write intermediate images
                                                                                       like the re-normalised inputs,
                                                                                       denoised images, the registered
                                                                                       images, etc.
               /test_scores.npy:                                                     - if name_xfm_dir and labels_dir
                                                                                       are given this is 3 row matrix
                                                                                       with errors in angle, shifts
                                                                                       and Dice scores.

    :param path_main_model: path of the saved model for the main network.
    :param main_data_dir: path of directory containing the subdirectory of all time series to process
    :param name_image_dir: name of subfolder with image in each time series directory. This needs to be consistent
    across all time series directory.
    :param net_type: type of feature extractor as defined in predict.py
    :param image_size: size of the testing input data. If some images in image_dir are not of this size,
    they will be zero-padded and cropped to fit image_size.
    :param name_xfm_dir: (optional) ame as name_image_dir but for ground truth rigid transforms, given as 4x4 numpy
    matrices in homogeneous coordinates. These are optional, and are only used to compute evaluation metric for the
    mean error for angle and shift prediction.
    :param name_labels_dir: (optional) same as name_image_dir but for masks. Masks are optional, and if provided, they
    will be used to compute Dice scores after registration as a complementary evaluation metric.
    :param min_perc: (optional) During training, image intensities are normalised in [0,1] with min-max normalisation.
    min_perc is the minimum percentile that will correspond to 0 after normalisation. Default is 0.01.
    :param max_perc: (optional) same as above but for maximum value. Default is 99.99.
    :param mean_05: whether to further re-center the average intensity to 0.5 after normalisation. Default is False.
    :param main_n_channels: (optional) number of output channels of the feature extractor. Default is 32.
    :param main_n_levels: (optional) number of resolution levels (only used if 'conv' is in net_type).
    :param main_n_conv: (optional) number of convolution per resolution level (only used if 'conv' is in net_type).
    :param main_n_feat: (optional) number of initial feature maps after the first convolution (only used if 'conv' is in
    net_type).
    :param main_feat_mult: (optional) feature multiplier after each max pooling (only used if 'conv' is in net_type).
    :param main_kernel_size: (optional) size of convolutional kernels  (only used if 'conv' is in net_type).
    :param main_last_activation: (optional) last non-linearity (only used if 'conv' is in net_type).
    :param closed_form_algo: (optional) type of the closed-form algorithm to use. Can be 'numerical' (default)
    or 'analytical', which implements the strategy used in KeyMorph.
    :param path_denoiser_model: (optional) path of the saved model for the denoising network. Default is None, where
    no denoising is performed.
    :param denoiser_n_levels: (optional) number of resolution levels for the denoising UNet. Default is 4.
    :param denoiser_n_conv: (optional) number of convolution per resolution level for the UNet. Default is 2.
    :param denoiser_n_feat: (optional) number of initial feature maps after the first convolution for the UNet.
    Default is 32.
    :param denoiser_feat_mult: (optional) feature multiplier after each max pooling for the UNet. Default is 1.
    :param denoiser_kernel_size: (optional) size of convolutional kernels for the UNet. Default is 3.
    :param denoiser_rm_top_skip_connection: (optional) whether to remove the top skip connection. Default is 1.
    :param denoiser_predict_residual: (optional) whether to add a residual connection between the input and last layer.
    :param recompute: (optional) whether to recompute the results if these are already saved at the given location.
    """

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size
    use_denoiser = True if ((path_denoiser_model != '') & (path_denoiser_model is not None)) else False

    net = None

    list_time_series = sorted([f for f in os.listdir(main_data_dir) if os.path.isdir(os.path.join(main_data_dir, f))])
    for time_series in list_time_series:
        series_dir = os.path.join(main_data_dir, time_series)

        results_dir = os.path.join(series_dir, 'equitrack')

        # create result directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'predicted_transforms', 'predicted_transforms_4x4_matrix'), exist_ok=True)
        path_results = os.path.join(results_dir, 'test_scores.npy')
        path_rotations = os.path.join(results_dir, 'predicted_transforms', 'predicted_rotation_angles.npy')
        path_translations = os.path.join(results_dir, 'predicted_transforms', 'predicted_translation_shifts.npy')

        if (not os.path.isfile(path_rotations)) | (not os.path.isfile(path_translations)) | \
                ((not os.path.isfile(path_results)) & (name_labels_dir is not None)) | recompute:

            # test loader
            testing_subj_dict = build_subject_dict_series(series_dir, name_image_dir, name_labels_dir)
            testing_xfm_dict = build_xfm_dict_series(series_dir, name_xfm_dir) if name_xfm_dir is not None else None
            test_dataset = loaders.loader_time_series(subj_dict=testing_subj_dict,
                                                      dict_xfm=testing_xfm_dict,
                                                      min_perc=min_perc,
                                                      max_perc=max_perc,
                                                      mean_05=mean_05,
                                                      return_masks=(name_labels_dir is not None),
                                                      resize=image_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
            list_of_test_frames = list()
            for list_of_tuples in testing_subj_dict.values():
                for j in range(1, len(list_of_tuples)):
                    list_of_test_frames.append(os.path.basename(list_of_tuples[j][0]))

            # prepare network
            if net is None:

                # initialise feature extractor
                net = networks.Archi(name=net_type,
                                     input_shape=image_size,
                                     n_out_chan=main_n_channels,
                                     n_levels=main_n_levels,
                                     n_conv=main_n_conv,
                                     n_feat=main_n_feat,
                                     feat_mult=main_feat_mult,
                                     kernel_size=main_kernel_size,
                                     last_activation=main_last_activation,
                                     return_inputs=use_denoiser,
                                     closed_form_algo=closed_form_algo).to(device)
                net.load_state_dict(torch.load(path_main_model, map_location=torch.device(device))['net_state_dict'])

                # prepend denoiser if necessary
                if use_denoiser > 0:
                    denoiser = networks.DenoiserWrapper(n_levels=denoiser_n_levels,
                                                        n_conv=denoiser_n_conv,
                                                        n_feat=denoiser_n_feat,
                                                        feat_mult=denoiser_feat_mult,
                                                        kernel_size=denoiser_kernel_size,
                                                        rm_top_skip_connection=denoiser_rm_top_skip_connection,
                                                        predict_residual=denoiser_predict_residual,
                                                        postprocess_outputs=True).to(device)
                    state_dict = torch.load(path_denoiser_model, map_location=torch.device(device))['net_state_dict']
                    state_dict_2 = dict()
                    for k, v in state_dict.items():
                        state_dict_2['denoiser.' + k] = v
                    denoiser.load_state_dict(state_dict_2)
                    net = torch.nn.Sequential(denoiser, net).to(device)

            # test loop
            net.eval()
            list_scores = []
            list_rotations = []
            list_translations = []
            for i, batch in enumerate(test_loader):

                # initialise inputs
                moving = batch['scan_moving'].to(device)
                fixed = batch['scan_fixed'].to(device)
                aff = batch['aff'].to(device)

                # predict transformation
                if use_denoiser:
                    xfm, denoised_moving, denoised_fixed = net.forward((moving, fixed))
                else:
                    xfm = net.forward((moving, fixed))
                    denoised_moving = denoised_fixed = None

                # apply transformation
                grid_xfm = aff_to_field(xfm, image_size, invert_affine=True)
                moved = torch.moveaxis(interpolate(torch.moveaxis(moving, 1, -1), grid_xfm, 1, 'linear'), -1, 1)

                # save images
                list_tensors = [moving, fixed, moved]
                list_names = ['moving', 'fixed', 'moved']
                if use_denoiser:
                    list_tensors += [denoised_moving, denoised_fixed]
                    list_names += ['moving_denoised', 'fixed_denoised']
                for tens, name in zip(list_tensors, list_names):
                    path = os.path.join(results_dir, 'test_images', name, list_of_test_frames[i])
                    save_volume(tens.cpu().detach().numpy().squeeze(), aff.cpu().detach().numpy().squeeze(), None, path)
                del tens

                # save transforms
                angles = matrix_to_angles(xfm[:, :3, :3], 'XYZ').cpu().detach().numpy().squeeze() * 180 / np.pi
                list_rotations.append(angles)
                list_translations.append(xfm[:, :3, 3].cpu().detach().numpy().squeeze())
                np.save(os.path.join(results_dir, 'predicted_transforms', 'predicted_transforms_4x4_matrix',
                                     list_of_test_frames[i].replace('nii.gz', 'npy')),
                        xfm.cpu().detach().numpy().squeeze())

                # evaluation
                tmp_list_scores = list()

                # compute transform similarity
                if name_xfm_dir is not None:
                    xfm_gt = batch['xfm'].to(device)
                    err_R = losses.rotation_matrix_to_angle_loss(xfm_gt[:, :3, :3], xfm[:, :3, :3]).item()
                    err_T = losses.translation_loss(xfm_gt[:, :3, 3], xfm[:, :3, 3]).item()
                    tmp_list_scores += [err_R, err_T]
                    del xfm_gt, err_R, err_T

                # compute Dice score and save masks
                if name_labels_dir:
                    mask_moving = batch['mask_moving'].to(device)
                    mask_fixed = batch['mask_fixed'].to(device)
                    mask_moved = interpolate(torch.moveaxis(mask_moving, 1, -1), grid_xfm, 1, 'nearest')
                    mask_moved = torch.moveaxis(mask_moved, -1, 1)
                    test_dice = losses.dice(mask_fixed, mask_moved).mean().item()
                    tmp_list_scores.append(test_dice)
                    aff = aff.cpu().detach().numpy().squeeze()
                    for tens, name in zip([mask_moving, mask_fixed, mask_moved],
                                          ['moving_mask', 'fixed_mask', 'moved_mask']):
                        path = os.path.join(results_dir, 'test_images', name, list_of_test_frames[i])
                        save_volume(tens.cpu().detach().numpy().squeeze(), aff, None, path)
                    del mask_moving, mask_fixed, mask_moved, test_dice, tens

                # save scores
                if tmp_list_scores:
                    list_scores.append(tmp_list_scores)

                # flush cuda memory
                del moving, fixed, aff, xfm, denoised_moving, denoised_fixed, grid_xfm, moved
                torch.cuda.empty_cache()

            # write scores
            np.save(path_rotations, np.array(list_rotations).T)
            np.save(path_translations, np.array(list_translations).T)
            if list_scores:
                np.save(path_results, np.array(list_scores).T)
