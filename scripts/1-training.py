"""
This first script shows how to train the framework for rigid transform estimation.

This framework has two parts: a feature extractor, and a rigid registration estimator. Both can be implemented different
ways (see below).

Importantly, we train this framework on simulated pairs: we first randomly select an anchor image,
which is then spatially deformed twice with rigid transforms to give us a fixed and a moving image with known transform
between the two.

Furthermore, the two training inputs can be separately augmented with different intensity transforms.

All parameters below are explained in details. None of the paths provided here exist, it's just an example :)
"""

from equitrack.training import training

# INPUTS
# the training script doesn't assume any particular structure in the data organisation. Yet, it needs two mandatory
# image folders: one for training and one for validation. Images need to be in nifty format.
training_im_dir = '/data/training/images'
val_im_dir = '/data/validation/images'
# the framework is fully differentiable and can be trained by optimising different losses. Among those losses, one can
# use Dice scores between teh masks of moved and fixed images. These masks are provided with the training_lab_dir
# parameter. This needs to be a folder with masks corresponding to the given images and sorted in the same order.
# However, this is not the solution that we implemented in our paper, since we obtain better results with an
# unsupervised image loss alone.
training_lab_dir = None
# every n epochs, the model is being validated on a separate folder, by computing the accuracy of the predicted
# transforms w.r.t. known ground truth transforms. But one can also add another validation metric by computing Dice
# scores between moved and fixed masks. So, here again we can provide
val_lab_dir = '/data/validation/labels'
# Finally, we need to provide a directory where all the intermediate models and validation scores will be saved.
results_dir = '/data/results_training_rigid_registration'

# ARCHITECTURE
# we first need to specify how to implement the two parts of our framework, which is done with the net_type parameter.
# The feature extractor can either be:
#    - an equivariant network (use 'se3' in name).
#    - or a conv net, which can be a UNet ('conv_unet') or an encoder ('conv_encoder')
# The rigid registration can either:
#    - collapse each feature map onto its center of mass, which gives us two point clouds. These are then registered
#       with a svd-based closed-form algorithm. Use 'svd' in the name.
#    - a direct regression based on densely connected conv layers ('dense'). This is incompatible with 'se3', and
#       'unet', as the dimension of the first densely connected layer would explode.
# The feature extractor can either process the two images to register in parallel (default), or as concatenated inputs
# ('concat'). However concatenated inputs are not compatible with svd, since the latter needs two point clouds.
# In summary net_type can be one of the following: se3_svd, conv_unet_svd, conv_encoder_dense, conv_encoder_dense_concat
# In our paper we show that se3_svd is superior to all other architectures for rigid motion tracking
net_type = 'se3_svd'
# Here, the architecture of the equivariant network is fixed, except for the number of output channels for the feature
# extractor.
n_channels = 64
# Therefore there's a bunch of parameters that we leave unspecified here, since those are only used for the other
# architectures:
n_levels = 0
n_conv = 0
n_feat = 0
feat_mult = 0
kernel_size = 0
last_activation = ''
# Finally, there are two options for the closed-form algorithm: one analytical solution implemented in KeyMorph that
# assumes perfect correspondence between the two point clouds to register (closed_form_algo = 'analytical'), but here
# we use another closed-form solution that gives the optimal solution to minimise distances between corresponding pairs
# of points after registration
closed_form_algo = 'numerical'

# PREPROCESSING
image_size = 96  # size of the images for training, which will be obtained by zero-padding and cropping
# input images are rescaled in [0,1] with min-max normalisation. Here the estimate robust minimum and maximum with given
# percentiles, which become the effective min and max.
min_perc = 0.01
max_perc = 99.99
# After re-normalisation, we can further make sure that the average intensity of strictly positive voxels is 0.5. We did
# not choose this option, but people might find it useful.
mean_05 = False

# augmentation
rotation_range = 90   # range for the rotations, which will be drawn in [-rotation_range, +rotation_range]
shift_range = 20      # same for translations
max_noise_std = 0.05  # maximum standard deviation of the Gaussian noise to apply (higher = stronger).
max_bias_std = 0.3    # maximum standard deviation of the bias field corruption to apply (higher = stronger).
bias_scale = 0.06     # scale of the bias field (lower = smoother).
gamma_std = 0.2       # maximum value of the gamma-exponentiation for histogram shifting (higher = stronger).

# Finally, some learning parameters
# Our framework is fully differentiable and can be trained with different losses: unsupervised l2 loss between the moved
# and fixed images, supervised Dice loss (don't forget to specify training_lab_dir in this case), supervised l2 loss
# between the estimated and ground truth transforms. Here we chose to only use the image loss, which gave us slightly
# better results, than any other combination.
batch_size = 4                  # change this to fit your GPU
learning_rate = 1e-5
n_epochs = 100000               # this is much larger than what we need in reality, so we stop this way earlier
validate_every_n_epoch = 100
weight_image_loss = 1
weight_dice_loss = 0
weight_xfm_loss = 0

training(training_im_dir=training_im_dir,
         val_im_dir=val_im_dir,
         results_dir=results_dir,
         net_type=net_type,
         image_size=image_size,
         training_lab_dir=training_lab_dir,
         val_lab_dir=val_lab_dir,
         rotation_range=rotation_range,
         shift_range=shift_range,
         mean_05=mean_05,
         min_perc=min_perc,
         max_perc=max_perc,
         max_noise_std=max_noise_std,
         max_bias_std=max_bias_std,
         bias_scale=bias_scale,
         gamma_std=gamma_std,
         n_channels=n_channels,
         n_levels=n_levels,
         n_conv=n_conv,
         n_feat=n_feat,
         feat_mult=feat_mult,
         kernel_size=kernel_size,
         last_activation=last_activation,
         closed_form_algo=closed_form_algo,
         batch_size=batch_size,
         learning_rate=learning_rate,
         n_epochs=n_epochs,
         validate_every_n_epoch=validate_every_n_epoch,
         weight_xfm_loss=weight_xfm_loss,
         weight_image_loss=weight_image_loss,
         weight_dice_loss=weight_dice_loss)
