"""This script is just to show how to use the denoiser on its own once it is trained. This is of no real use since
the denoiser will be integrated to our framework in another script.
"""

from equitrack.predict_denoiser import predict

# inputs (Images need to be in nifty format.)
path_model = '/data/results_denoiser/best_val_loss.pth'  # this is the path of the trained denoiser
testing_image_dir = '/data/testing/images'               # this is the path to the testing images to denoise

# this script will write two kinds of outputs: the denoised images, but also the input images to the network.
# The latter are saved because there's a bit of preprocessing (image size but also normalisation in [0,1]), so saving
# the preprocessed data is important to see the real effect of the denoiser.
results_dir_inputs = '/data/results_testing_denoiser/inputs'
results_dir_denoised = '/data/results_testing_denoiser/denoised'

# preprocessing (must be the same as in 2-training_denoiser.py)
image_size = 96     # resize testing images to this size by zero-padding and cropping
min_perc = 0.01     # percentile for robust minimum for min-max normalisation of the intensities in [0,1]
max_perc = 99.99    # same for maximum percentile
mean_05 = False     # further intensity preprocessing by centering positive intensities around 0.5

# architecture of the denoising UNet (same os the training architecture)
n_levels = 4                 # number of resolution levels in the UNet
n_conv = 2                   # number of convolutional layers for rach resolution level
n_feat = 32                  # number of feature maps for the very first layer of the network
feat_mult = 1                # multiplies the number of feature maps at each level. 1 keeps the same number of features
kernel_size = 3              # size of the convolutional masks
rm_top_skip_connection = 1   # number of top skip connections to remove
predict_residual = True      # whether to add a residual connection between the input and the last layer.

predict(path_model=path_model,
        image_dir=testing_image_dir,
        results_dir_inputs=results_dir_inputs,
        results_dir_denoised=results_dir_denoised,
        image_size=image_size,
        min_perc=min_perc,
        max_perc=max_perc,
        mean_05=mean_05,
        n_levels=n_levels,
        n_conv=n_conv,
        n_feat=n_feat,
        feat_mult=feat_mult,
        kernel_size=kernel_size,
        rm_top_skip_connection=rm_top_skip_connection,
        predict_residual=predict_residual)
