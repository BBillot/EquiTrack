import os
import re
import numpy as np
import torch.utils.data
from shutil import copy2

from equitrack import networks
from equitrack import loaders
from equitrack.losses import image_loss
from equitrack.utils import build_subject_dict, save_volume

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def training(training_im_dir,
             val_im_dir,
             results_dir,
             image_size,
             rotation_range=90,
             shift_range=20,
             min_perc=0.01,
             max_perc=99.99,
             mean_05=False,
             max_noise_std=0.05,
             max_bias_std=0.3,
             bias_scale=0.06,
             gamma_std=0.2,
             n_levels=4,
             n_conv=2,
             n_feat=32,
             feat_mult=1,
             kernel_size=3,
             rm_top_skip_connection=1,
             predict_residual=True,
             batch_size=4,
             learning_rate=1e-5,
             n_epochs=100000,
             validate_every_n_epoch=100,
             resume=False):
    """
    This function trains a UNet to denoise input data that have been corrupted with Gaussian noise, histogram shift and
    bias field.
    :param training_im_dir: path of the directory with training images
    :param val_im_dir: path of the directory with validation images
    :param results_dir: path of the directory where all training models, losses, and intermediate results will be saved.
    :param image_size: size of the training input data. If some images in training/validation dir are not of this size,
    they will be zero-padded and cropped to fit image_size.
    This can be an integer (same size in all three directions), or a sequence of length 3.
    :param rotation_range: (optional) range for the rotation augmentation. Must be a float/integer where angles will be sampled
    uniformly in [-rotation_range, +rotation_range]
    :param shift_range: (optional) same as above but for translations.
    :param min_perc: (optional) During training, image intensities are normalised in [0,1] with min-max normalisation.
    min_perc is the minimum percentile that will correspond to 0 after normalisation. Defaults to 0.01.
    :param max_perc: (optional) same as above but for maximum value. Defaults to 99.99.
    :param mean_05: (optional) whether to further re-center the average intensity to 0.5 after normalisation
    (default is False).
    :param max_noise_std: (optional) maximum standard deviation of the Gaussian noise to apply (higher = stronger).
    Default is 0.05.
    :param max_bias_std: (optional) maximum standard deviation of the bias field corruption to apply
    (higher = stronger). Default is 0.3.
    :param bias_scale: (optional) scale of the bias field (lower = smoother). Default is 0.06.
    :param gamma_std: (optional) maximum value of the gamma-exponentiation for histogram shifting (higher = stronger).
    Default is 0.2.
    :param n_levels: (optional) number of resolution levels for the UNet. Default is 4.
    :param n_conv: (optional) number of convolution per resolution level for the UNet. Default is 2.
    :param n_feat: (optional) number of initial feature maps after the first convolution for the UNet. Default is 32.
    :param feat_mult: (optional) feature multiplier after each max pooling for the UNet. Default is 1.
    :param kernel_size: (optional) size of convolutional kernels for the UNet. Default is 3.
    :param rm_top_skip_connection: (optional) whether to remove the top skip connection. Default is 1.
    :param predict_residual: (optional) whether to add a residual connection between the input and the last layer.
    Default is true.
    :param batch_size: (optional) number of examples per training batch. Default is 4, but change this to fit GPU memory
    :param learning_rate: (optional) learning rate for the Adam optimiser. Default is 10^-5.
    :param n_epochs: (optional) number of training epochs (set this to a high value to ensure convergence). Default is
    100000, which is waaaayyyy beyond convergence (I always stop training before this).
    :param validate_every_n_epoch: (optional) frequency of the validation step for model evaluation. Default is 100.
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
    training_subj_dict = build_subject_dict(training_im_dir, None)
    train_dataset = loaders.loader_denoiser(subj_dict=training_subj_dict,
                                            augm_params=augment_params,
                                            seed=1919)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # validation loader
    print('create validation loader...', flush=True)
    val_subj_dict = build_subject_dict(val_im_dir, None)
    val_dataset = loaders.loader_denoiser(subj_dict=val_subj_dict,
                                          augm_params=augment_params,
                                          validator_mode=True,
                                          seed=1919)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    list_val_scans = list(val_subj_dict.keys())

    # initialise feature extractor
    print('initialise architecture...', flush=True)
    net = networks.UNet(n_levels=n_levels,
                         n_conv=n_conv,
                         n_feat=n_feat,
                         feat_mult=feat_mult,
                         kernel_size=kernel_size,
                         rm_top_skip_connection=rm_top_skip_connection,
                         predict_residual=predict_residual)
    net = net.to(device)

    # initialise optimizer
    print('initialise optimizer...\n', flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load last file if necessary
    last_epoch = 0
    if resume:
        previous_files = sorted([p for p in os.listdir(models_dir) if re.sub('\D', '', p) != ''])
        if len(previous_files) > 0:
            print(f'loading from {previous_files[-1]}', flush=True)
            checkpoint = torch.load(os.path.join(models_dir, previous_files[-1]), map_location=torch.device(device))
            net.load_state_dict(checkpoint['net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']

    # Training loop
    best_val_loss = 1e9
    list_scores = []
    for epoch in range(last_epoch, n_epochs):
        print('Epoch', epoch, flush=True)

        net.train()
        epoch_train_loss = 0
        for i, batch in enumerate(train_loader):

            # initialise inputs
            noisy_scan = batch["noisy_scan"].to(device)
            clean_scan = batch["clean_scan"].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                # get predictions
                denoised_scan = net.forward(noisy_scan)

                # compute loss
                train_loss = image_loss(clean_scan, denoised_scan)

                # backpropagation
                train_loss.backward(retain_graph=True)
                optimizer.step()

            # print iteration info
            epoch_train_loss += train_loss.item() * batch_size
            print('iteration:{}/{}  loss:{:.5f}'.format(i + 1, len(train_dataset), train_loss.item()), flush=True)

            # flush cuda memory
            del noisy_scan, clean_scan, denoised_scan, train_loss
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
            for j, batch in enumerate(val_loader):

                # initialise inputs
                noisy_scan = batch['noisy_scan'].to(device)
                clean_scan = batch["clean_scan"].to(device)

                # get predictions
                denoised_scan = net.forward(noisy_scan)

                # compute image loss
                denoised_scan = denoised_scan * (noisy_scan > 0).to(dtype=denoised_scan.dtype)
                val_loss = image_loss(clean_scan, denoised_scan)
                epoch_val_loss += val_loss.item()

                # save examples
                for tens, name in zip([noisy_scan, clean_scan, denoised_scan], ['noisy', 'clean', 'denoised']):
                    path = os.path.join(results_dir, 'val_images', '%05d' % epoch, name, list_val_scans[j] + '.nii.gz')
                    save_volume(tens.cpu().detach().numpy().squeeze(), None, None, path)

                # flush cuda memory
                del noisy_scan, clean_scan, denoised_scan, val_loss, tens
                torch.cuda.empty_cache()

            # save validation scores
            epoch_val_loss = epoch_val_loss / len(val_dataset)
            print('Epoch:{}  Train Loss:{:.5f}  Val Loss:{:.5f}'.format(
                epoch, epoch_train_loss, epoch_val_loss) + '\n', flush=True)
            list_scores.append([epoch, epoch_train_loss, epoch_val_loss])
            np.save(os.path.join(results_dir, 'val_scores.npy'), np.array(list_scores))

            # save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                copy2(os.path.join(models_dir, '%05d.pth' % epoch), os.path.join(results_dir, 'best_val_loss.pth'))
                with open(os.path.join(results_dir, 'best_val_loss.txt'), 'w') as f:
                    f.write('epoch:%d   val loss:%f' % (epoch, best_val_loss))

    del net
