"""This script performs rigid motion tracking in time series of nifty images by registering all time frames to the very
first one of the time series. This script has a strict requirement in terms of the structure of the provided data."""

from equitrack.predict_time_series import predict

# inputs
path_main_model = '/data/results_rigid_registration/best_val_loss.pth'  # path of the rigid registration model
# This function expects the input data to be organised as follows:
# main_data_dir/time_series_1/name_image_dir: image_time_frame_0.nii.gz
#                                             image_time_frame_1.nii.gz
#                                             ...
#             (optionally)    name_label_dir: label_time_frame_0.nii.gz
#                                             label_time_frame_1.nii.gz
#                                             ...
#             (optionally)    name_xfm_dir:   gt_4x4_transform_matrix_1_to_0.npy
#                                             ...
#              /time_series_2/name_image_dir: image_time_frame_0.nii.gz
#                                             image_time_frame_1.nii.gz
#                                             ...
#             (optionally)    name_label_dir: label_time_frame_0.nii.gz
#                                             label_time_frame_1.nii.gz
#                                             ...
#             (optionally)    name_xfm_dir:   gt_4x4_transform_matrix_1to0.npy
#                                             ...
# Here we provide the main data directory, which includes all the time series that we want to process. Each time series
# thn has an identical structure for its folder, with the same subfolder name for images (and optionally masks and
# transforms). The code will loop over all subfolders in main_data_dir, so it should only contain time-series subfolders
main_data_dir = '/data/testing/time_series'
name_image_dir = 'images'        # this needs to be the same name for all time series folder
name_labels_dir = 'masks'        # if you don't have this replace with None
name_xfm_dir = 'gt_transforms'   # if you don't have this replace with None

# results will automatically be saved in main_data_dir/time_series_1/equitrack with the same structure as in 4-predict.py:
# main_data_dir/time_series_1/equitrack/predicted_transforms/predicted_transforms_4x4_matrix/*.npy
#                                                        /predicted_rotation_angles.npy
#                                                        /predicted_translation_shifts.npy
#                   (optionally)    /test_scores.npy
#                                   /test_images/moving
#                                   /test_images/fixed (all the same as this is always the 1st frame of the series)
#                                   /test_images/moved
#                                   /test_images/moving_denoised
#                                   /test_images/fixed_denoised (all the same as this is always the 1st frame)
#                   (optionally)    /test_images/mask_moving
#                   (optionally)    /test_images/mask_fixed (all the same as this is always the 1st frame)
#                   (optionally)    /test_images/mask_moved
# the outputs are the same as for tutorial 4, except that we now save them for every time series separately.

# preprocessing
image_size = 96     # resize testing images to this size by zero-padding and cropping
min_perc = 0.01     # percentile for robust minimum for min-max normalisation of the intensities in [0,1]
max_perc = 99.99    # same for maximum percentile
mean_05 = False     # further intensity preprocessing by centering positive intensities around 0.5

# architecture main network (must be the same as in 1-training.py)
net_type = 'se3_svd'
main_n_channels = 64
main_n_levels = 0
main_n_conv = 0
main_n_feat = 0
main_feat_mult = 0
main_kernel_size = 0
main_last_activation = ''
closed_form_algo = 'numerical'

# to activate the denoiser network (this considerably increases the robustness of the overall framework!!!!) you need to
# specify the path to a trained model
path_denoiser_model = '/data/results_denoiser/best_val_loss.pth'
# the rest of the parameters describe the architecture of teh denoiser (must be the same as in 2-training_denoiser.py)
denoiser_n_levels = 4
denoiser_n_conv = 2
denoiser_n_feat = 32
denoiser_feat_mult = 1
denoiser_kernel_size = 3
denoiser_rm_top_skip_connection = 1
denoiser_predict_residual = True

# whether to recompute the time-series that have already been fully processed
recompute = True

predict(path_main_model=path_main_model,
        main_data_dir=main_data_dir,
        name_image_dir=name_image_dir,
        net_type=net_type,
        image_size=image_size,
        name_labels_dir=name_labels_dir,
        name_xfm_dir=name_xfm_dir,
        min_perc=min_perc,
        max_perc=max_perc,
        mean_05=mean_05,
        main_n_channels=main_n_channels,
        main_n_levels=main_n_levels,
        main_n_conv=main_n_conv,
        main_n_feat=main_n_feat,
        main_feat_mult=main_feat_mult,
        main_kernel_size=main_kernel_size,
        main_last_activation=main_last_activation,
        closed_form_algo=closed_form_algo,
        path_denoiser_model=path_denoiser_model,
        denoiser_n_levels=denoiser_n_levels,
        denoiser_n_conv=denoiser_n_conv,
        denoiser_n_feat=denoiser_n_feat,
        denoiser_feat_mult=denoiser_feat_mult,
        denoiser_kernel_size=denoiser_kernel_size,
        denoiser_rm_top_skip_connection=denoiser_rm_top_skip_connection,
        denoiser_predict_residual=denoiser_predict_residual,
        recompute=recompute)
