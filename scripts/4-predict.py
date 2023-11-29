"""This script shows how to use the complete framework for estimation of rigid transform between pairs of images.
It uses the feature extractor and the rigid transform estimator, possibly with an appended denoiser.
Ground truth transforms can also be provided to compute angle and shift errors in the predicted transforms.
"""

from energi.predict import predict

# INPUTS (Images need to be in nifty format.)
path_main_model = '/data/results_training_rigid_registration/best_val_loss.pth'  # path of the rigid registration model
# Since this script registers PAIRS of data, we need to specify two directories of corresponding data pairs, sorted in
# the same order.
testing_moving_im_dir = '/data/testing/images_moving'
testing_fixed_im_dir = '/data/testing/images_fixed'

# This is the directory where the predicted transforms and the moved images will be saved. Specifically the result
# folder will have the following structure:
# results_dir/predicted_transforms/predicted_transforms_4x4_matrix/*.npy   - 4x4 numpy arrays with the estimated rigid
#                                                                            transform for every pair
#                                 /predicted_rotation_angles.npy           - 3xN_pairs numpy array which is a summary of
#                                                                            all predicted rotation angles
#                                 /predicted_translation_shifts.npy        - same but for translation shifts
#            /test_images/inputs_moving                                    - preprocessed (size, normalisation) moving
#                                                                            images
#            /test_images/inputs_fixed                                     - preprocessed fixed images
#            /test_images/inputs_moved                                     - moved images
#            /test_images/denoised_moving                                  - denoised moving images if denoiser is
#                                                                            activated (see below)
#            /test_images/denoised_fixed                                   - same but for fixed images
results_dir = '/data/pair_registration_results/'

# If one has access to binary masks (0/1 integers), these can also be provided to compute Dice scores between the fixed
# and moved masks. Moreover, if one has access to ground truth transforms, these can also be given to compute errors in
# the predicted rotation angles and translation shifts. In this case ew will have the following additional outputs
# results_dir/test_scores.npy which is a 3xN_pairs numpy array. The first row is the mean angle error (average across
# the three directions), the 2nd row is the shift error, and the third row are the Dice scores.
testing_moving_lab_dir = '/data/testing/masks_moving'
testing_fixed_lab_dir = '/data/testing/masks_fixed'
testing_dir_xfm = '/data/testing/xfm_1to2'

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

predict(path_main_model=path_main_model,
        results_dir=results_dir,
        testing_moving_im_dir=testing_moving_im_dir,
        testing_fixed_im_dir=testing_fixed_im_dir,
        net_type=net_type,
        image_size=image_size,
        testing_moving_lab_dir=testing_moving_lab_dir,
        testing_fixed_lab_dir=testing_fixed_lab_dir,
        testing_dir_xfm=testing_dir_xfm,
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
        denoiser_predict_residual=denoiser_predict_residual)
