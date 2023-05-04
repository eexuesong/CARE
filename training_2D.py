from __future__ import print_function, unicode_literals, absolute_import, division
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline

import tifffile
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.data import RawData, create_patches, no_background_patches
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.models import Config, CARE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit TensorFlow to first GPU.

# limit_gpu_memory(fraction=1/2)
r"""
The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
"""


class training_2d():
    def __init__(self, training_dir, input_folder, truth_folder, model_name, show_flag=False):
        self.training_dir = training_dir
        self.input_folders = [input_folder]
        self.truth_folder = truth_folder
        self.model_name = model_name
        self.training_data_name = os.path.join(training_dir, model_name + '.npz')

        self.show_flag = show_flag
        self.raw_data = None
        self.patch_size = None
        self.patch_number = None

        self.patch_cal()
        self.data_gen(show_flag=self.show_flag)
        self.training(show_flag=self.show_flag)

    def patch_cal(self):
        r"""
        Calculate the training data patch size for CARE
        :return:
        """
        gt_dir = os.path.join(self.training_dir, self.truth_folder)
        gt_filenames = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.tif')]
        r"""
        gt_filenames = []
        gt_dir = os.path.join(training_dir, truth_folder)
        for f in os.listdir(gt_dir):
            file_name = os.path.join(gt_dir, f)
            if os.path.isfile(file_name):
                gt_filenames.append(os.path.join(gt_dir, f))
        """

        # Iterate all files and check whether they have the same dimension, data type and axes.
        ny_gt = []
        nx_gt = []
        pixel_number_gt = []
        data_gt = []
        axes_gt = []
        for filename in gt_filenames:
            # Get information about the image stack in the TIFF file without reading any image data:
            tif = tifffile.TiffFile(filename)
            series = tif.series[0]  # get shape and dtype of first image series
            (ny, nx) = series.shape
            ny_gt.append(ny)
            nx_gt.append(nx)
            pixel_number_gt.append(ny * nx)
            data_gt.append(series.dtype)
            axes_gt.append(series.axes)

        if not all(data_type == data_gt[0] for data_type in data_gt):
            raise UserWarning("Ground truth files do not have the same data type.")
        if not all(axes_order == axes_gt[0] for axes_order in axes_gt):
            raise UserWarning("Ground truth files do not have the same axes order.")

        r"""
        As a general rule, use a patch size that is a power of two along all axes, or which is at least divisible by 8.
        """
        if all(ny == ny_gt[0] for ny in ny_gt) and all(nx == nx_gt[0] for nx in nx_gt):
            print("GT training data size [ny, nx]: ({0}, {1})".format(ny_gt[0], nx_gt[0]))

            if ny_gt[0] <= 64:
                patch_sizeY = ny_gt[0]
            else:
                patch_sizeY = 64

            if nx_gt[0] <= 64:
                patch_sizeX = nx_gt[0]
            else:
                patch_sizeX = 64

            self.patch_size = (patch_sizeY, patch_sizeX)  # y, x (64, 64)
            self.patch_number = int(ny_gt[0] * nx_gt[0] / (patch_sizeY * patch_sizeX))
        else:
            print("GT training files do not have the same dimension. Patch size will be calculated from the minimum pixels.")
            ny_min = min(ny_gt)
            nx_min = min(nx_gt)

            if ny_min <= 64:
                patch_sizeY = ny_min
            else:
                patch_sizeY = 64

            if nx_min <= 64:
                patch_sizeX = nx_min
            else:
                patch_sizeX = 64

            self.patch_size = (patch_sizeY, patch_sizeX)
            self.patch_number = int(sum(pixel_number_gt) / (patch_sizeY * patch_sizeX))

        print("Patch size: {0} x {1}".format(self.patch_size[0], self.patch_size[1]))
        print("Patch number: {0}".format(self.patch_number))
        print()

    def data_gen(self, show_flag=False):
        r"""
        Generate training data for CARE
        :return:
        """
        # We first need to create a RawData object, which defines how to get the pairs of low/high SNR stacks and the
        # semantics of each axis (e.g. which one is considered a color channel, etc.).
        self.raw_data = RawData.from_folder(
            basepath=self.training_dir,
                # str – Base folder that contains sub-folders with images.
            source_dirs=self.input_folders,
                # list or tuple – List of folder names relative to basepath that contain the source images (e.g., with low SNR).
            target_dir=self.truth_folder,
                # str – Folder name relative to basepath that contains the target images (e.g., with high SNR).
            axes='YX',
                # str – Semantics of axes of loaded images (assumed to be the same for all images).
        )
        r"""
        It is important that you correctly set the axes of the raw images,
        e.g. to 'CYX' for 2D images with a channel dimension before the two lateral dimensions. 'ZYX' for 3D image stacks.
        Image Axes - X: columns, Y: rows, Z: planes, C: channels, T: frames/time, (S: samples/images)
        Note:
            Source and target images must be well-aligned to obtain effective CARE networks.    
        """

        X, Y, XY_axes = create_patches(
            raw_data=self.raw_data,
                # RawData – Object that yields matching pairs of raw images.
            patch_size=self.patch_size,
                # tuple – Shape of the patches to be extracted from raw images.
                # Must be compatible with the number of dimensions and axes of the raw images.
                # As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
            patch_filter=no_background_patches(0),
                # By default, patches are sampled from non-background regions
                # i.e. that are above a relative threshold that can be given in the function below.
                # We will disable this if most image regions in the dataset already contain foreground pixels and thus set the threshold to 0.
            n_patches_per_image=self.patch_number,
                # int - Number of patches to be sampled/extracted from each raw image pair.
            save_file=self.training_data_name
                # str or None – File name to save training data to disk in .npz format.
                # If None, data will not be saved.
        )
        r"""
        Returns a tuple (X, Y, axes) with the normalized extracted patches from all (transformed) raw images and their axes.
        X is the array of patches extracted from source images.
        Y is the array of corresponding target patches.
        The shape of X and Y is as follows: (n_total_patches, n_channels, ...), e.g. (21400, 1, 64, 64, 64)
        For single-channel images, n_channels will be 1.
        """

        assert X.shape == Y.shape
        print("shape of X,Y =", X.shape)  # (21400, 1, 64, 64)
        print("axes  of X,Y =", XY_axes)  # 'SCYX'
        print()

        if show_flag:
            # This shows some of the generated patch pairs
            # (odd rows: source, even rows: target)
            plt.figure(figsize=(16, 4))
            plt.suptitle('Top row: source, Bottom row: target')
            sl = slice(0, 8), 0
            plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
            plt.show()

    def training(self, show_flag=False):
        r"""
        Training data
        :return:
        """
        # region Load training data, and split them into 95% actual training data and 5% validation data.
        (X, Y), (X_val, Y_val), axes = load_training_data(
            file=self.training_data_name,
            validation_split=0.05,  # float – Fraction of images to use as validation set during training.
            verbose=True)
        print()
        r"""
        X: patches from source images, input images (e.g., with low SNR).
        Y: patches from target images, ground truth images (e.g., with high SNR)
        (X_train, Y_train): tuple(numpy.ndarray, numpy.ndarray) - training data sets, e.g. (21400, 64, 64, 64, 1)
        (X_val, Y_val):     tuple(numpy.ndarray, numpy.ndarray) - validation data sets, e.g. (2140, 64, 64, 64, 1)
        axes: str - the axes of the input images, e.g. 'SZYXC'
        Note:
            (1) After load_training_data, the channel axis will be changed from 'SCZYX' to 'SZYXC'.
            (2) The tuple of validation data will be None if validation_split = 0.
        """
        c = axes_dict(axes)['C']  # from axes string to dict
        # axes_dict(axes) = {'S': 0; 'T': None; 'C': 4; 'Z': 1; 'Y': 2; 'X': 3}
        # axes_dict(axes)['C'] = 4
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]  # n_channel_in, n_channel_out = 1, 1

        # Show: this shows some of the maximum projections of the generated patch pairs
        # (odd rows: source, even rows: target)
        if show_flag:
            plt.figure(figsize=(12, 5))
            plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
            plot_some(X_val[:5], Y_val[:5])
            plt.show()
        # endregion

        # region CARE model
        r"""
        Before we construct the actual CARE model, we have to define its configuration via a Config object, which includes
            * parameters of the underlying neural network,
            * the learning rate,
            * the number of parameter updates per epoch,
            * the loss function, and
            * whether the model is probabilistic or not.
        The defaults should be sensible in many cases, so a change should only be necessary if the training process fails.
        """
        config = Config(
            axes=axes,                      # str – Axes of the neural network (channel axis optional).
            n_channel_in=n_channel_in,      # int – Number of channels of given input image.
            n_channel_out=n_channel_out,    # int – Number of channels of predicted output image.
            unet_kern_size=3,   # int - Parameter kern_size of csbdeep.nets.common_unet(). Default: 5 if n_dim==2 else 3
            train_epochs=100,   # int - Number of training epochs. Default: 100
            train_steps_per_epoch=100       # int - Number of parameter update steps per epoch. Default: 400
        )
        r"""
        Note: use a very small number (e.g. train_steps_per_epoch=10) for immediate feedback.
        Increased this number considerably (e.g. train_steps_per_epoch=400) to obtain a well-trained model.
        """
        print(config)
        vars(config)
        print()

        # Standard CARE network for image restoration and enhancement.
        r""" We now create a CARE model with the chosen configuration: """
        model = CARE(
            config=config,
                # csbdeep.models.Config or None – Valid configuration of CARE network.
                # Will be saved to disk as JSON (config.json).
                # If set to None, will be loaded from disk (must exist).
            name=self.model_name,
                # str or None – Model name. Uses a timestamp if set to None (default).
            basedir=self.training_dir
                # str – Directory that contains (or will contain) a folder with the given model name.
                # Use None to disable saving (or loading) any data to (or from) disk (regardless of other parameters).
        )

        # We can inspect the created neural network:
        model.keras_model.summary()
        # endregion

        # region Training
        r"""
        Training the model will likely take some time.
        """
        start_time = time.time()
        history = model.train(X, Y, validation_data=(X_val, Y_val))
        end_time = time.time()
        training_time = end_time - start_time
        training_sec = training_time
        training_hour = int(training_sec // 3600)
        training_sec %= 3600
        training_min = int(training_sec // 60)
        training_sec = int(training_sec % 60)
        print("Total training time: {0:d}:{1:02d}:{2:02d} ({3:0.1f} seconds)".format(training_hour, training_min,
                                                                                     training_sec, training_time))
        print()
        # Plot final training history (available in TensorBoard during training):
        print(sorted(list(history.history.keys())))
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
        # endregion

        # region Evaluation
        if show_flag:
            plt.figure(figsize=(12, 7))
            _P = model.keras_model.predict(X_val[:5])
            if config.probabilistic:
                _P = _P[..., :(_P.shape[-1] // 2)]
            plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5)
            plt.suptitle('5 example validation patches\n'
                         'top row: input (source),  '
                         'middle row: target (ground truth),  '
                         'bottom row: predicted from source')
        # endregion


if __name__ == "__main__":
    print('CARE training.\n')

    # region Folders for training
    training_dir = r''
    input_folder = '2d_low'
    truth_folder = '2d_GT'
    model_name = ''
    # endregion

    training_2d(training_dir=training_dir, input_folder=input_folder, truth_folder=truth_folder, model_name=model_name,
                show_flag=True)