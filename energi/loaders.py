import math
import copy
import torch
import numpy as np
import torch.utils.data
import numpy.random as npr

from energi.utils import preprocess
from energi.augmenters import SpatialAugmenter, IntensityAugmenter


class loader_rxfm(torch.utils.data.IterableDataset):
    """Implements a generator that will give training example at each minibatch to the rigid transform estimator
    network in pytorch."""

    def __init__(self,
                 subj_dict,
                 augm_params=None,
                 validator_mode=False,
                 return_masks=False,
                 return_clean_images=False,
                 seed=None):
        """
        :param subj_dict: dictionary of the form {'im basename': [path_im, path_mask]} as built by the function
        utils.build_subject_dict. the 'path_mask' entry is optional.
        :param augm_params: dictionary containing values for the preprocessing and augmentation:
            augment_params = {'resize': image_size,             # resize input images to this size with padding/cropping
                              'rotation_range': rotation_range, # maximum rotation angle for augmentation (in degrees)
                              'shift_range': shift_range,       # maximum shift for augmentation (in voxels)
                              'crop_size': None,                # randomly crop examples to size, None=no cropping
                              'flip': False,                    # randomly flip examples
                              'min_perc': min_perc,     # clip intensities to given percentile before rescaling to [0,1]
                              'max_perc': max_perc,             # same but for top clipping
                              'mean_05': mean_05,               # recenter intensities after rescaling to have 0.5 mean
                              'max_noise_std': max_noise_std,   # maximum standard deviation for the Gaussian noise
                              'max_bias_std': max_bias_std,     # maximum std. dev for the bias filed corruption
                              "bias_scale": bias_scale,         # scale of the bias field (lower = smoother)
                              "gamma_std": gamma_std}           # std dev for random exponentiation (higher = stronger)
        :param validator_mode: whether to build a loader for validation data. In this case, each validation example
        will be augmented the same way for each validation step
        :param return_masks: whether to return masks as additional volumes at each minibatch. These must be given in
        subj_dict.
        :param return_clean_images: whether to return the input example before intensity augmentation as
        additional volumes at each minibatch (useful to compute image loss).
        :param seed: numpy seed to keep the augmentation the same across training runs.
        """

        # input data
        self.subj_dict = subj_dict
        self.list_of_subjects = list(self.subj_dict.keys())
        self.n_samples = len(self.list_of_subjects)
        self.iter_idx = 0

        # outputs
        self.return_masks = return_masks
        self.return_clean_images = return_clean_images

        # initialise resize/rescale functions
        resize = augm_params["resize"]
        self.preproc_func = lambda x: preprocess(x,
                                                 normalise=True,
                                                 min_perc=augm_params["min_perc"],
                                                 max_perc=augm_params["max_perc"],
                                                 mean_05=augm_params["mean_05"],
                                                 resize=resize)
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize)

        # numpy seed
        self.validator_mode = validator_mode
        self.rng = npr.RandomState(seed)

        # load, resize, rescale images/labels (still numpy with size [H, W, D, C])
        self.samples = {}  # {'im basename': [im, mask]}
        for subj in self.list_of_subjects:
            self.samples[subj] = self.load_sample(self.subj_dict[subj])

        # get augmentation parameters if in validation mode (still numpy)
        augment_discrete = []
        if self.validator_mode:
            bias_sample_size = [math.ceil(size * augm_params["bias_scale"]) for size in resize]
            for _ in self.list_of_subjects:
                r = self.rng.uniform(-augm_params["rotation_range"], augm_params["rotation_range"], 3).tolist()
                t = self.rng.uniform(-augm_params["shift_range"], augm_params["shift_range"], 3).tolist()
                if augm_params["max_noise_std"] > 0:
                    noise = self.rng.normal(0, self.rng.uniform(high=augm_params["max_noise_std"]), resize)
                else:
                    noise = None
                if augm_params["max_bias_std"] > 0:
                    bias = self.rng.normal(0, self.rng.uniform(high=augm_params["max_bias_std"]), bias_sample_size)
                else:
                    bias = None
                if augm_params["gamma_std"] > 0:
                    gamma = self.rng.normal(0, augm_params["gamma_std"])
                else:
                    gamma = None
                augment_discrete.append({"rotation": r,
                                         "translation": t,
                                         "noise_field": noise,
                                         "bias_field": bias,
                                         "gamma": gamma})

        # initialise spatial/intensity augmenters
        self.spatial_augmenter = SpatialAugmenter(list_of_xfm_params=augment_discrete,
                                                  rotation_range=augm_params["rotation_range"],
                                                  shift_range=augm_params["shift_range"],
                                                  crop_size=augm_params["crop_size"],  # None
                                                  flip=augm_params["flip"],  # False
                                                  return_affine=True,
                                                  normalise=True,
                                                  mean_05=augm_params["mean_05"],
                                                  seed=seed)
        self.intensity_augmenter = IntensityAugmenter(list_of_params=augment_discrete,
                                                      max_noise_std=augm_params["max_noise_std"],
                                                      max_bias_std=augm_params["max_bias_std"],
                                                      bias_scale=augm_params["bias_scale"],
                                                      gamma_std=augm_params["gamma_std"],
                                                      mean_05=augm_params["mean_05"],
                                                      seed=seed)

        # output format params
        self.output_names = ["scan_moving", "scan_fixed", "xfm"]
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]
        if self.return_clean_images:
            self.output_names += ["clean_scan_moving", "clean_scan_fixed"]

    def load_sample(self, sample_tuple):
        sample = [self.preproc_func(sample_tuple[0])]  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def __next__(self):

        if self.iter_idx >= self.n_samples:
            self.iter_idx = 0  # reset at the end of every epoch
            raise StopIteration

        # load a random [frame, mask] from a random subject
        idx = np.random.choice(self.n_samples) if not self.validator_mode else self.iter_idx % self.n_samples
        frame_mask_moving = self.samples[self.list_of_subjects[idx]]  # frame size [H, W, D, C]
        frame_mask_fixed = copy.deepcopy(frame_mask_moving)

        # spatial augment, outputs numpy matrices in all cases
        if self.validator_mode:
            xfm_moving = np.eye(4)  # no transformation
            frame_mask_fixed, xfm_fixed = self.spatial_augmenter.predefined_transform(idx, *frame_mask_fixed)
        else:
            frame_mask_moving, xfm_moving = self.spatial_augmenter.random_transform(*frame_mask_moving)
            frame_mask_fixed, xfm_fixed = self.spatial_augmenter.random_transform(*frame_mask_fixed)
        xfm = xfm_fixed.astype('float32') @ np.linalg.inv(xfm_moving.astype('float32'))

        # get clean frames (ie not yet augmented for intensity)
        clean_frame_moving = frame_mask_moving[0]
        clean_frame_fixed = frame_mask_fixed[0]

        # intensity augment
        if self.validator_mode:
            frame_mask_fixed[0] = self.intensity_augmenter.predefined_transform(idx, frame_mask_fixed[0])
        else:
            frame_mask_moving[0] = self.intensity_augmenter.random_transform(frame_mask_moving[0])
            frame_mask_fixed[0] = self.intensity_augmenter.random_transform(frame_mask_fixed[0])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32),
                   xfm.astype(np.float32)]
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[1], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[1], 3, 0).astype(np.float32)]
        if self.return_clean_images:
            outputs += [np.rollaxis(clean_frame_moving, 3, 0).astype(np.float32),
                        np.rollaxis(clean_frame_fixed, 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        self.iter_idx += 1
        return output_dict

    def __iter__(self):
        self.iter_idx = 0  # reset at the start of every epoch
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()


class loader_denoiser(torch.utils.data.IterableDataset):
    """Implements a generator to feed the denoising network in pytorch with training examples."""

    def __init__(self,
                 subj_dict,
                 augm_params=None,
                 validator_mode=False,
                 seed=None):
        """
        :param subj_dict: dictionary of the form {'im basename': [path_im]} as built by the function
        utils.build_subject_dict
        :param augm_params: dictionary containing values for the preprocessing and augmentation:
            augment_params = {'resize': image_size,             # resize input images to this size with padding/cropping
                              'rotation_range': rotation_range, # maximum rotation angle for augmentation (in degrees)
                              'shift_range': shift_range,       # maximum shift for augmentation (in voxels)
                              'crop_size': None,                # randomly crop examples to size, None=no cropping
                              'flip': False,                    # randomly flip examples
                              'min_perc': min_perc,     # clip intensities to given percentile before rescaling to [0,1]
                              'max_perc': max_perc,             # same but for top clipping
                              'mean_05': mean_05,               # recenter intensities after rescaling to have 0.5 mean
                              'max_noise_std': max_noise_std,   # maximum standard deviation for the Gaussian noise
                              'max_bias_std': max_bias_std,     # maximum std. dev for the bias filed corruption
                              "bias_scale": bias_scale,         # scale of the bias field (lower = smoother)
                              "gamma_std": gamma_std}           # std dev for random exponentiation (higher = stronger)
        :param validator_mode: whether to build a loader for validation data. In this case, each validation example
        will be augmented the same way for each validation step
        :param seed: numpy seed to keep the augmentation the same across training runs.
        """

        # input data
        self.subj_dict = subj_dict
        self.list_of_subjects = list(self.subj_dict.keys())

        # batch/epoch params
        self.n_samples = len(self.list_of_subjects)
        self.iter_idx = 0

        # initialise resize/rescale functions
        resize = augm_params["resize"]
        self.preproc_func = lambda x: preprocess(x,
                                                 normalise=True,
                                                 min_perc=augm_params["min_perc"],
                                                 max_perc=augm_params["max_perc"],
                                                 mean_05=augm_params["mean_05"],
                                                 resize=resize)

        # numpy seed
        self.validator_mode = validator_mode
        self.rng = npr.RandomState(seed)

        # load, resize, rescale images (still numpy with size [H, W, D, C])
        self.samples = {}  # {'im basename': [im]}
        for subj in self.list_of_subjects:
            self.samples[subj] = [self.load_sample(self.subj_dict[subj])]

        # get augmentation parameters if in validation mode (still numpy)
        augment_discrete = []
        if self.validator_mode:
            bias_sample_size = [math.ceil(size * augm_params["bias_scale"]) for size in resize]
            for _ in self.list_of_subjects:
                r = self.rng.uniform(-augm_params["rotation_range"], augm_params["rotation_range"], 3).tolist()
                t = self.rng.uniform(-augm_params["shift_range"], augm_params["shift_range"], 3).tolist()
                if augm_params["max_noise_std"] > 0:
                    noise = self.rng.normal(0, self.rng.uniform(high=augm_params["max_noise_std"]), resize)
                else:
                    noise = None
                if augm_params["max_bias_std"] > 0:
                    bias = self.rng.normal(0, self.rng.uniform(high=augm_params["max_bias_std"]), bias_sample_size)
                else:
                    bias = None
                if augm_params["gamma_std"] > 0:
                    gamma = self.rng.normal(0, augm_params["gamma_std"])
                else:
                    gamma = None
                augment_discrete.append({"rotation": r,
                                         "translation": t,
                                         "noise_field": noise,
                                         "bias_field": bias,
                                         "gamma": gamma})

        # initialise spatial/intensity augmenters
        self.spatial_augmenter = SpatialAugmenter(list_of_xfm_params=augment_discrete,
                                                  rotation_range=augm_params["rotation_range"],
                                                  shift_range=augm_params["shift_range"],
                                                  crop_size=augm_params["crop_size"],  # None
                                                  flip=augm_params["flip"],  # False
                                                  return_affine=False,
                                                  normalise=True,
                                                  mean_05=augm_params["mean_05"],
                                                  seed=seed)
        self.intensity_augmenter = IntensityAugmenter(list_of_params=augment_discrete,
                                                      max_noise_std=augm_params["max_noise_std"],
                                                      max_bias_std=augm_params["max_bias_std"],
                                                      bias_scale=augm_params["bias_scale"],
                                                      gamma_std=augm_params["gamma_std"],
                                                      mean_05=augm_params["mean_05"],
                                                      seed=seed)

        # output format params
        self.output_names = ["noisy_scan", "clean_scan"]

    def load_sample(self, sample_tuple):
        return self.preproc_func(sample_tuple)  # [H, W, D, C]

    def __next__(self):

        if self.iter_idx >= self.n_samples:
            self.iter_idx = 0  # reset at the end of every epoch
            raise StopIteration

        # load a random frame
        idx = np.random.choice(self.n_samples) if not self.validator_mode else self.iter_idx % self.n_samples
        frame_mask = self.samples[self.list_of_subjects[idx]]  # frame size [H, W, D, C]

        # spatial augment, outputs numpy matrices in all cases
        if self.validator_mode:
            frame_mask = self.spatial_augmenter.predefined_transform(idx, *frame_mask)
        else:
            frame_mask = self.spatial_augmenter.random_transform(*frame_mask)

        # get clean frames (ie not yet augmented for intensity)
        clean_frame = frame_mask[0]

        # intensity augment
        if self.validator_mode:
            frame_mask[0] = self.intensity_augmenter.predefined_transform(idx, frame_mask[0])
        else:
            frame_mask[0] = self.intensity_augmenter.random_transform(frame_mask[0])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(clean_frame, 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        self.iter_idx += 1
        return output_dict

    def __iter__(self):
        self.iter_idx = 0  # reset at the start of every epoch
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()


class loader_testing(torch.utils.data.IterableDataset):
    """Implements a generator to feed the framework for rigid transform estimation between 2 images with testing data.
    This is the case where we have pairs that have been simulated beforehand, so here we take as inputs:
    moving images, fixed images and gt transforms"""

    def __init__(self,
                 subj_dict_moving,
                 subj_dict_fixed,
                 min_perc,
                 max_perc,
                 mean_05,
                 resize,
                 return_masks,
                 dict_xfm=None):
        """
        :param subj_dict_moving: dictionary of the form {'im basename': [path_im, path_mask]} as built by the function
        utils.build_subject_dict. This is for the moving images. The 'path_mask' entry is optional.
        :param subj_dict_fixed: same as above but for the fixed images
        :param min_perc: clip intensities to given percentile before rescaling to [0,1]
        :param max_perc: same but for top clipping
        :param mean_05: recenter intensities after rescaling to have 0.5 mean
        :param resize: resize input images to this size with padding/cropping, list of image shape [H, W, D]
        :param return_masks: whether to return masks as additional volumes at each minibatch. These must be given in
        subj_dict_1 and subj_dict_2.
        :param dict_xfm: (optional) dictionary of the form {'im basename': path_gt_xfm}, where path_gt_xfm contains GT transforms
        to go from image 1 to image 2.
        """

        # input data
        self.subj_dict_moving = subj_dict_moving
        self.subj_dict_fixed = subj_dict_fixed
        self.dict_xfm = dict_xfm
        self.return_masks = return_masks
        self.list_of_subjects = list(self.subj_dict_moving.keys())
        self.n_samples = len(self.list_of_subjects)
        self.subject_idx = -1

        # initialise resize/rescale functions
        self.preproc_func = lambda x: preprocess(x,
                                                 normalise=True,
                                                 min_perc=min_perc,
                                                 max_perc=max_perc,
                                                 mean_05=mean_05,
                                                 resize=resize)
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize)

        # output format params
        self.output_names = ["scan_moving", "scan_fixed"]
        if self.dict_xfm is not None:
            self.output_names += ["xfm"]
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]

    def load_sample(self, sample_tuple):
        sample = [self.preproc_func(sample_tuple[0])]  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def __next__(self):

        self.subject_idx += 1
        if self.subject_idx >= self.n_samples:
            raise StopIteration

        # load data for first and current datapoints
        frame_mask_moving = self.load_sample(self.subj_dict_moving[self.list_of_subjects[self.subject_idx]])
        frame_mask_fixed = self.load_sample(self.subj_dict_fixed[self.list_of_subjects[self.subject_idx]])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32)]
        if self.dict_xfm is not None:
            outputs += [np.load(self.dict_xfm[self.list_of_subjects[self.subject_idx]]).astype(np.float32)]
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[1], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[1], 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        return output_dict

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()


class loader_testing_denoiser(torch.utils.data.IterableDataset):
    """Implements a generator to feed the denoising network with testing data."""

    def __init__(self, subj_dict, min_perc, max_perc, mean_05, resize):
        """
        :param subj_dict dictionary of the form {'im basename': [path_im, path_mask]} as built by the function
        utils.build_subject_dict.
        :param min_perc: clip intensities to given percentile before rescaling to [0,1]
        :param max_perc: same but for top clipping
        :param mean_05: recenter intensities after rescaling to have 0.5 mean
        :param resize: resize input images to this size with padding/cropping, list of image shape [H, W, D]
        """

        # input data
        self.subj_dict = subj_dict
        self.list_of_subjects = list(self.subj_dict.keys())
        self.n_samples = len(self.list_of_subjects)
        self.subject_idx = -1

        # initialise resize/rescale functions
        self.preproc_func = lambda x: preprocess(x,
                                                 normalise=True,
                                                 min_perc=min_perc,
                                                 max_perc=max_perc,
                                                 mean_05=mean_05,
                                                 resize=resize)

    def load_sample(self, sample_tuple):
        return self.preproc_func(sample_tuple[0])  # [H, W, D, C]

    def __next__(self):

        self.subject_idx += 1
        if self.subject_idx >= self.n_samples:
            raise StopIteration

        # load data for first and current datapoints
        frame = self.load_sample(self.subj_dict[self.list_of_subjects[self.subject_idx]])

        # group outputs in dict
        output_dict = {'scan': torch.tensor(np.rollaxis(frame, 3, 0).astype(np.float32))}  # [C, H, W, D]
        return output_dict

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()


class loader_time_series(torch.utils.data.IterableDataset):
    """Implements a generator for feeding the testing data to the framework for rigid transform estimation between
    2 images. This is the case of real time-series, where the GT transforms might not been known.
    This takes as inputs lists of time frames, where we register every time frame to the first of the series"""

    def __init__(self,
                 subj_dict,
                 min_perc,
                 max_perc,
                 mean_05,
                 resize,
                 return_masks,
                 dict_xfm=None):
        """
        :param subj_dict: dictionary of the form built in utils.build_subject_dict_time_series:
        {'time_series_1_dir': [[image_0, label_0], [image_1, label_1], ...]}
        :param dict_xfm: dictionary of the form built in utils.build_xfm_dict_time_series:
        {'im basename': [path_xfm_1, path_xfm_2, ...]} which contain GT transforms from image i to image 0
        :param min_perc: clip intensities to given percentile before rescaling to [0,1]
        :param max_perc: same but for top clipping
        :param mean_05: recenter intensities after rescaling to have 0.5 mean
        :param resize: resize input images to this size with padding/cropping, list of image shape [H, W, D]
        """

        # input data
        self.subj_dict = subj_dict
        self.dict_xfm = dict_xfm
        self.return_masks = return_masks
        self.list_of_subjects = list(self.subj_dict.keys())
        self.subject_idx = 0
        self.frame_idx = 0

        # initialise resize/rescale functions
        self.preproc_func = lambda x: preprocess(x,
                                                 normalise=True,
                                                 min_perc=min_perc,
                                                 max_perc=max_perc,
                                                 mean_05=mean_05,
                                                 resize=resize,
                                                 return_aff=True)
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize)

        # output format params
        self.output_names = ["scan_moving", "scan_fixed", "aff"]
        if self.dict_xfm is not None:
            self.output_names.append("xfm")
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]

    def load_sample(self, sample_tuple):
        sample = list(self.preproc_func(sample_tuple[0]))  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def __next__(self):

        self.frame_idx += 1
        if self.frame_idx == len(self.subj_dict[self.list_of_subjects[self.subject_idx]]):
            if self.subject_idx < (len(self.list_of_subjects) - 1):
                self.subject_idx += 1
                self.frame_idx = 1
                print('')
            else:
                raise StopIteration

        # load data for first and current datapoints
        frame_mask_moving = self.load_sample(self.subj_dict[self.list_of_subjects[self.subject_idx]][self.frame_idx])
        frame_mask_fixed = self.load_sample(self.subj_dict[self.list_of_subjects[self.subject_idx]][0])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32),
                   frame_mask_moving[1]]                                        # aff to save the images
        if self.dict_xfm is not None:
            xfm_1to2 = np.load(self.dict_xfm[self.list_of_subjects[self.subject_idx]][self.frame_idx - 1])
            outputs += [xfm_1to2.astype(np.float32)]
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[2], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[2], 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        return output_dict

    def __iter__(self):
        return self

    def __len__(self):
        n_iterations = 0
        for subj in self.list_of_subjects:
            n_iterations += len(self.subj_dict[subj])
        return n_iterations

    def next(self):
        return self.__next__()
