import os
import torch.utils.data

from equitrack import loaders
from equitrack import networks
from equitrack.utils import build_subject_dict, save_volume

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def predict(path_model,
            image_dir,
            results_dir_inputs,
            results_dir_denoised,
            image_size,
            min_perc=0.01,
            max_perc=99.9,
            mean_05=False,
            n_levels=4,
            n_conv=2,
            n_feat=32,
            feat_mult=1,
            kernel_size=3,
            rm_top_skip_connection=1,
            predict_residual=True):
    """
    This function removes noise from input images as it has been trained to do in training_denoiser.
    :param path_model: path of the model to use for denoising.
    :param image_dir: path of the directory with testing images
    :param results_dir_inputs: path of the directory where inputs to the network will be saved. These can be different
    from the images in image_dir because of resizing and intensity normalisation.
    :param results_dir_denoised: path of the directory where denoised images will be saved.
    :param image_size: size of the testing input data. If some images in image_dir are not of this size,
    they will be zero-padded and cropped to fit image_size.
    :param min_perc: During training, image intensities are normalised in [0,1] with min-max normalisation. min_perc is
    the minimum percentile that will correspond to 0 after normalisation. Default is 0.01.
    :param max_perc: same as above but for maximum value. Default is 99.99.
    :param mean_05: whether to further re-center the average intensity to 0.5 after normalisation. Default is False.
    :param n_levels: number of resolution levels for the UNet. Default is 4.
    :param n_conv: number of convolution per resolution level for the UNet. Default is 2.
    :param n_feat: number of initial feature maps after the first convolution for the UNet. Default is 32.
    :param feat_mult: feature multiplier after each max pooling for the UNet. Default is 1.
    :param kernel_size: size of convolutional kernels for the UNet. Default is 3.
    :param rm_top_skip_connection: whether to remove the top skip connection. Default is 1.
    :param predict_residual: whether to add a residual connection between the input and the last layer.
    """

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size

    # create result directory
    os.makedirs(results_dir_inputs, exist_ok=True)
    os.makedirs(results_dir_denoised, exist_ok=True)

    # test loader
    testing_subj_dict = build_subject_dict(image_dir)
    list_subjects = list(testing_subj_dict.keys())
    test_dataset = loaders.loader_testing_denoiser(subj_dict=testing_subj_dict,
                                                   min_perc=min_perc,
                                                   max_perc=max_perc,
                                                   mean_05=mean_05,
                                                   resize=image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # initialise feature extractor
    net = networks.UNet(n_levels=n_levels,
                        n_conv=n_conv,
                        n_feat=n_feat,
                        feat_mult=feat_mult,
                        kernel_size=kernel_size,
                        rm_top_skip_connection=rm_top_skip_connection,
                        predict_residual=predict_residual)
    net = net.to(device)
    state_dict = torch.load(path_model, map_location=torch.device(device))['net_state_dict']
    net.load_state_dict(state_dict)
    net = torch.nn.Sequential(net, net).to(device)

    # test loop
    net.eval()
    for i, batch in enumerate(test_loader):

        # initialise inputs
        input_scan = batch['scan'].to(device)

        # predict transformation
        denoised_scan = net.forward(input_scan)

        # postprocess denoised image
        denoised_scan = torch.clamp(denoised_scan, min=0)
        denoised_scan = denoised_scan * (input_scan > 0).to(dtype=denoised_scan.dtype)
        min_1 = torch.amin(denoised_scan, dim=[1, 2, 3, 4], keepdim=True)
        max_1 = torch.amax(denoised_scan, dim=[1, 2, 3, 4], keepdim=True)
        denoised_scan = (denoised_scan - min_1) / (max_1 - min_1)

        # save images
        path_input = os.path.join(results_dir_inputs, list_subjects[i] + '.nii.gz')
        path_denoised = os.path.join(results_dir_denoised, list_subjects[i] + '.nii.gz')
        save_volume(input_scan.cpu().detach().numpy().squeeze(), None, None, path_input)
        save_volume(denoised_scan.cpu().detach().numpy().squeeze(), None, None, path_denoised)

        # flush cuda memory
        del input_scan, denoised_scan
        torch.cuda.empty_cache()
