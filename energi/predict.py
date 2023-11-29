import os
import numpy as np
import torch.utils.data
from pytorch3d.transforms import matrix_to_euler_angles

from energi import loaders
from energi import networks
import energi.losses as losses
from energi.utils import build_subject_dict, build_xfm_dict, aff_to_field, interpolate, save_volume

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings("ignore")


def predict(path_main_model,
            results_dir,
            testing_moving_im_dir,
            testing_fixed_im_dir,
            net_type,
            image_size,
            testing_moving_lab_dir=None,
            testing_fixed_lab_dir=None,
            testing_dir_xfm=None,
            min_perc=0.01,
            max_perc=99.99,
            mean_05=False,
            main_n_channels=64,
            main_n_levels=5,
            main_n_conv=1,
            main_n_feat=64,
            main_feat_mult=1,
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
            denoiser_predict_residual=True):
    """
    This function predicts rigid transforms between given pairs of fixed and moved images. The 'main' network consists
    in a feature extractor (an equivariant network or a CNN) and a rigid transform estimator (SVD-based closed-form
    algorithm or fully connected layers). In addition, one can prepend a denoising CNN to remove anatomically
    irrelevant features.

    The predicted transforms will be saved in the provided results_dir in two ways:
        1- results_dir/predicted_transforms/predicted_transforms_4x4_matrix/
            this folder will contain numpy arrays with predicted rigid transforms in homogeneous coordinates
        2- results_dir/predicted_transforms/predicted_rotation_angles.npy and predicted_translation_shifts.npy
            these are numpy matrices of shape 3xN_image_pairs with rotation and translations

    Finally, ground truth transforms can be given as well as binary masks for the moving and fixed images.
    In this case, we compute evaluation metrics which are saved under results_dir/test_scores.npy.
    This is a 3xN_image_pairs matrix where the rows correspond to 1) mean error in predicted angle 2) mean error in
    predicted translation 3) Dice score.

    :param path_main_model: path of the saved model for the main network.
    :param results_dir: path where all results will be saved.
    :param testing_moving_im_dir: path of the directory with all the MOVING images.
    :param testing_fixed_im_dir: path of the directory with all the FIXED images.
    :param net_type: type of feature extractor as defined in predict.py
    :param image_size: size of the testing input data. If some images in image_dir are not of this size,
    they will be zero-padded and cropped to fit image_size.
    :param testing_moving_lab_dir: (optional) path of the directory with all the MOVING masks. These are used to compute
    Dice for evaluation purposes. Not used by default.
    :param testing_fixed_lab_dir: (optional) path of the directory with all the FIXED masks. Not used by default.
    :param testing_dir_xfm: (optional) path of the directory with all the gt rigid transform. These must be 4x4 numpy
    arrays. These are used to compute rotation and translation errors.
    :param min_perc: During training, image intensities are normalised in [0,1] with min-max normalisation. min_perc is
    the minimum percentile that will correspond to 0 after normalisation. Default is 0.01.
    :param max_perc: same as above but for maximum value. Default is 99.99.
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
    """

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size
    use_denoiser = True if path_denoiser_model is not None else False
    if ((testing_moving_lab_dir is None) & (testing_fixed_lab_dir is not None)) | \
            ((testing_moving_lab_dir is not None) & (testing_fixed_lab_dir is None)):
        raise ValueError('testing_dir_lab_1 and testing_dir_lab_2 should either be both None or both given.')

    # create result directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'predicted_transforms', 'predicted_transforms_4x4_matrix'), exist_ok=True)
    path_results = os.path.join(results_dir, 'test_scores.npy')
    path_rotations = os.path.join(results_dir, 'predicted_transforms', 'predicted_rotation_angles.npy')
    path_translations = os.path.join(results_dir, 'predicted_transforms', 'predicted_translation_shifts.npy')

    # test loader
    testing_moving_subj_dict = build_subject_dict(testing_moving_im_dir, testing_moving_lab_dir)
    testing_fixed_subj_dict = build_subject_dict(testing_fixed_im_dir, testing_fixed_lab_dir)
    testing_xfm_dict = build_xfm_dict(testing_dir_xfm) if testing_dir_xfm is not None else None
    test_dataset = loaders.loader_testing(subj_dict_moving=testing_moving_subj_dict,
                                          subj_dict_fixed=testing_fixed_subj_dict,
                                          dict_xfm=testing_xfm_dict,
                                          min_perc=min_perc,
                                          max_perc=max_perc,
                                          mean_05=mean_05,
                                          return_masks=(testing_moving_lab_dir is not None),
                                          resize=image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    list_of_test_subjects = list(testing_moving_subj_dict.keys())

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
    if use_denoiser:
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
            path = os.path.join(results_dir, 'test_images', name, list_of_test_subjects[i] + '.nii.gz')
            save_volume(tens.cpu().detach().numpy().squeeze(), None, None, path)
        del tens

        # save transforms
        angles = matrix_to_euler_angles(xfm[:, :3, :3], 'XYZ').cpu().detach().numpy().squeeze() * 180 / np.pi
        list_rotations.append(angles)
        list_translations.append(xfm[:, :3, 3].cpu().detach().numpy().squeeze())
        np.save(os.path.join(results_dir, 'predicted_transforms', 'predicted_transforms_4x4_matrix',
                             list_of_test_subjects[i] + '.npy'),
                xfm.cpu().detach().numpy().squeeze())

        # evaluation
        tmp_list_scores = list()

        # compute transform similarity
        if testing_xfm_dict is not None:
            xfm_gt = batch['xfm'].to(device)
            err_R = losses.rotation_matrix_to_angle_loss(xfm_gt[:, :3, :3], xfm[:, :3, :3]).item()
            err_T = losses.translation_loss(xfm_gt[:, :3, 3], xfm[:, :3, 3]).item()
            tmp_list_scores += [err_R, err_T]
            del xfm_gt, err_R, err_T

        # compute Dice score and save masks
        if testing_moving_lab_dir is not None:
            mask_moving = batch['mask_moving'].to(device)
            mask_fixed = batch['mask_fixed'].to(device)
            mask_moved = torch.moveaxis(interpolate(torch.moveaxis(mask_moving, 1, -1), grid_xfm, 1, 'nearest'), -1, 1)
            test_dice = losses.dice(mask_fixed, mask_moved).mean().item()
            tmp_list_scores.append(test_dice)
            for tens, name in zip([mask_moving, mask_fixed, mask_moved], ['moving_mask', 'fixed_mask', 'moved_mask']):
                path = os.path.join(results_dir, 'test_images', name, list_of_test_subjects[i] + '.nii.gz')
                save_volume(tens.cpu().detach().numpy().squeeze(), None, None, path)
            del mask_moving, mask_fixed, mask_moved, test_dice, tens

        # save scores
        if tmp_list_scores:
            list_scores.append(tmp_list_scores)

        # flush cuda memory
        del moving, fixed, xfm, denoised_moving, denoised_fixed, grid_xfm, moved
        torch.cuda.empty_cache()

    # write scores
    np.save(path_rotations, np.array(list_rotations).T)
    np.save(path_translations, np.array(list_translations).T)
    if list_scores:
        np.save(path_results, np.array(list_scores).T)
