"""If implemented with an equivariant network, you will have seen by now that the previous framework is not able to
adapt to any of the intensity augmentation. Hence we propose here to train a denoiser that will be prepended to the
equivariant network at test-time to "denoise" the input images, such that those are roughly the same, up to the
unknown transform.
We do not train end-to-end with the denoiser and equivariant framework since, in practice we observe that the
equivariant network kills all the gradients, so none of these flow to the denoiser.

Here the denoiser is simply trained by removing simulated noise from real images. Hence we use a simple L2 loss between
the native image and the denoised version.
"""

from equitrack.training_denoiser import training

# inputs/outputs (Images need to be in nifty format.)
training_im_dir = '/data/training/images'  # training images
val_im_dir = '/data/validation/images'     # validation images
results_dir = '/data/results_denoiser'     # directory where models and validation scores will be saved.

# general
image_size = 96   # size of the training inputs, enforced by zero-padding and cropping
min_perc = 0.01   # percentile for robust minimum for min-max normalisation of the intensities in [0,1]
max_perc = 99.99  # same for maximum percentile
mean_05 = False   # further intensity preprocessing by centering positive intensities around 0.5

# noise corruption, this is the noise that the network will be trained to undo
max_noise_std = 0.05  # maximum standard deviation of the Gaussian noise to apply (higher = stronger).
max_bias_std = 0.3    # maximum standard deviation of the bias field corruption to apply (higher = stronger).
bias_scale = 0.06     # scale of the bias field (lower = smoother).
gamma_std = 0.2       # maximum value of the gamma-exponentiation for histogram shifting (higher = stronger).

# here we also perform some spatial augmentation to diversify teh training data
rotation_range = 90  # range for the rotations, which will be drawn in [-rotation_range, +rotation_range]
shift_range = 20     # same for translations

# architecture of the denoising UNet
n_levels = 4                 # number of resolution levels in the UNet
n_conv = 2                   # number of convolutional layers for rach resolution level
n_feat = 32                  # number of feature maps for the very first layer of the network
feat_mult = 1                # multiplies the number of feature maps at each level. 1 keeps the same number of features
kernel_size = 3              # size of the convolutional masks
predict_residual = True      # whether to add a residual connection between the input and the last layer.
# the following specifies the number of skip connections to remove, starting from the top level. This avoids to
# re-introduce noisy features at the latest levels of the network, close to the prediction.
rm_top_skip_connection = 1

# learning
batch_size = 4                  # change this to fit your GPU
learning_rate = 1e-5
n_epochs = 100000              # this is much larger than what we need in reality, so we stop this way earlier
validate_every_n_epoch = 100

training(training_im_dir=training_im_dir,
         val_im_dir=val_im_dir,
         results_dir=results_dir,
         image_size=image_size,
         rotation_range=rotation_range,
         shift_range=shift_range,
         min_perc=min_perc,
         max_perc=max_perc,
         max_noise_std=max_noise_std,
         max_bias_std=max_bias_std,
         bias_scale=bias_scale,
         gamma_std=gamma_std,
         mean_05=mean_05,
         n_levels=n_levels,
         n_conv=n_conv,
         n_feat=n_feat,
         feat_mult=feat_mult,
         kernel_size=kernel_size,
         rm_top_skip_connection=rm_top_skip_connection,
         predict_residual=predict_residual,
         batch_size=batch_size,
         learning_rate=learning_rate,
         n_epochs=n_epochs,
         validate_every_n_epoch=validate_every_n_epoch)
