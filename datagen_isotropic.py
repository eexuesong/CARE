import os
import numpy as np
import math
import ImageJ_formatted_TIFF
from scipy.ndimage import zoom, gaussian_filter1d
from skimage import transform

class datagen_isotropic():
    def __init__(self, training_dir, input_folder, truth_folder, resolution=0.05, spacing=0.125, voxel_size=0.05,
                 use_header_flag=False, test_mode=False, sigma_x=2.6, sigma_z=1.3, resampling_flag=True):
        self.training_dir = training_dir
        self.input_folder = input_folder
        self.truth_folder = truth_folder

        # region Create two folders for training data of input and GT
        self.input_dir = os.path.join(self.training_dir, self.input_folder)
        try:
            if not os.path.exists(self.input_dir):
                os.makedirs(self.input_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.input_dir)
        else:
            print("Successfully created the directory %s " % self.input_dir)

        self.truth_dir = os.path.join(self.training_dir, self.truth_folder)
        try:
            if not os.path.exists(self.truth_dir):
                os.makedirs(self.truth_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.truth_dir)
        else:
            print("Successfully created the directory %s " % self.truth_dir)
        # endregion

        self.voxel_size = voxel_size
        self.test_mode = test_mode
        self.resampling_flag = resampling_flag
        self.training_filenames = [f for f in os.listdir(self.training_dir) if f.endswith('.tif')]

        if self.test_mode:
            file_num = 1  # For testing purposes only, usually to adjust sigma_x and sigma_z
        else:
            file_num = len(self.training_filenames)

        # Iterate all files and check whether they have the same voxels after scaling.
        r"""
        For someone who gives me image datasets with non-uniform dimensions or lateral / axial pixel size.
        This situation usually happens when biologists acquired images on PMT-based system like 
        confocal, airyscan or two-photon whose pixel size is arbitrary.
        """
        nz_training = []
        ny_training = []
        nx_training = []

        for i in range(0, file_num):
            # Get information about the image stack in the TIFF file without reading any image data:
            training_filename = os.path.join(self.training_dir, self.training_filenames[i])
            header = ImageJ_formatted_TIFF.get_header(filename=training_filename)

            if use_header_flag:
                nz_scale = round(header.images * header.spacing / voxel_size)
                ny_scale = round(header.ImageLength * header.resolution / voxel_size)
                nx_scale = round(header.ImageWidth * header.resolution / voxel_size)
            else:
                nz_scale = round(header.images * spacing / voxel_size)
                ny_scale = round(header.ImageLength * resolution / voxel_size)
                nx_scale = round(header.ImageWidth * resolution / voxel_size)

            nz_training.append(nz_scale)
            ny_training.append(ny_scale)
            nx_training.append(nx_scale)

        if all(nz == nz_training[0] for nz in nz_training) and all(ny == ny_training[0] for ny in ny_training) and all(
                nx == nx_training[0] for nx in nx_training):
            pass
        else:
            print("Training files do not have the same dimensions after scaling.")

        nz_max = max(nz_training)
        ny_max = max(ny_training)
        nx_max = max(nx_training)
        print(
            "Maximum image size after scaling in training datasets [nz, ny, nx]: ({0}, {1}, {2})".format(nz_max, ny_max,
                                                                                                         nx_max))
        print()

        # Processing.
        for i in range(0, file_num):
            print("processing... {0}".format(self.training_filenames[i]))

            training_filename = os.path.join(self.training_dir, self.training_filenames[i])
            (training_data, header) = ImageJ_formatted_TIFF.ReadTifStack(filename=training_filename)
            if not use_header_flag:
                header.resolution = resolution
                header.spacing = spacing

            (nz, ny, nx) = training_data.shape
            print("Image size [nz, ny, nx]: ({0}, {1}, {2})".format(nz, ny, nx))

            # Scale voxel size and swap z, y axes to get raw data (usually not saved)
            raw_data = self.scale_voxel_swap_axes(image_data=training_data, lateral_pixel_size=header.resolution,
                                                  axial_pixel_size=header.spacing, voxel_size=voxel_size)

            # 1d Gaussian filter along current y-axis (previous z-axis) to suppress scaling artifacts in frequency domain.
            r"""
            Note by Xuesong:
                We now use "order=3" in skimage.transform.resize which already suppresses high-frequency repetitions.
                So sigma_z is unnecessary.
            """
            # GT_data = gaussian_filter1d(raw_data, sigma_z, axis=1)
            GT_data = raw_data
            GT_data[GT_data <= 0] = 0

            if self.test_mode:
                GT_data = self.apodize(image_data=GT_data)

            (nz, ny, nx) = GT_data.shape
            print("Image size after resize and swap axes [nz, ny, nx]: ({0}, {1}, {2})".format(nz, ny, nx))

            r"""
            Pad along z, y, x axes to match the maximum voxels of all training datasets.
            If all training datasets have the same voxels after scaling, this part of code does not work.
            """
            pad_z = round((ny_max - nz) / 2)
            pad_y = round((nz_max - ny) / 2)
            pad_x = round((nx_max - nx) / 2)
            pad_width = (pad_z, ny_max - nz - pad_z), (pad_y, nz_max - ny - pad_y), (pad_x, nx_max - nx - pad_x)
            GT_data_pad = np.pad(array=GT_data, pad_width=pad_width, mode="constant", constant_values=0)

            # Write Ground Truth image data
            GT_filename = os.path.join(self.truth_dir, self.training_filenames[i])
            ImageJ_formatted_TIFF.WriteTifStack(numpy_array=GT_data_pad, filename=GT_filename, resolution=self.voxel_size,
                                                spacing=self.voxel_size)
            print("Image size after padding [nz, ny, nx]: ({0}, {1}, {2})".format(ny_max, nz_max, nx_max))

            # 1d Gaussian filter along x-axis to create a blurred but x-z isotropic input image data.
            input_data = gaussian_filter1d(GT_data, sigma_x, axis=2)

            # Down-sampling & up-sampling resize along x axis after blurring.
            if self.resampling_flag:
                # Deprecated: sometimes it will change nx by 1 after down-sampling & up-sampling.
                # downsampling_x = header.spacing / self.voxel_size
                # input_data = zoom(input=input_data, zoom=(1, 1, 1/downsampling_x), order=3, mode="reflect")
                # input_data = zoom(input=input_data, zoom=(1, 1, downsampling_x), order=3, mode="reflect")

                nx_downsampling = round(nx * self.voxel_size / header.spacing)
                input_data = transform.resize(image=input_data, output_shape=(nz, ny, nx_downsampling), order=3,
                                              mode="reflect", preserve_range=True)
                input_data = transform.resize(image=input_data, output_shape=(nz, ny, nx), order=3, mode="reflect",
                                              preserve_range=True)
            input_data[input_data <= 0] = 0

            r"""
            Pad along z, y, x axes to match the maximum voxels of all training datasets.
            If all training datasets have the same voxels after scaling, this part of code does not work.
            """
            input_data = np.pad(array=input_data, pad_width=pad_width, mode="constant", constant_values=0)

            # Write input image data
            input_filename = os.path.join(self.input_dir, self.training_filenames[i])
            ImageJ_formatted_TIFF.WriteTifStack(numpy_array=input_data, filename=input_filename,
                                                resolution=self.voxel_size, spacing=self.voxel_size)
            print()

    def scale_voxel_swap_axes(self, image_data, lateral_pixel_size=0.05, axial_pixel_size=0.125, voxel_size=0.05):
        # Change the image voxel size to be the same (e.g. 50 nm) in x, y, z
        (nz, ny, nx) = image_data.shape
        nz_new = round(nz * axial_pixel_size / voxel_size)
        ny_new = round(ny * lateral_pixel_size / voxel_size)
        nx_new = round(nx * lateral_pixel_size / voxel_size)
        image_data = transform.resize(image=image_data, output_shape=[nz_new, ny_new, nx_new], order=3, mode="reflect",
                                      preserve_range=True)
        r"""
        Note by Xuesong: if using "order=3" in skimage.transform.resize, sigma_z is unnecessary.
        """
        # zoom_x = lateral_pixel_size / voxel_size
        # zoom_y = lateral_pixel_size / voxel_size
        # zoom_z = axial_pixel_size / voxel_size
        # image_data = zoom(image_data, (zoom_z, zoom_y, zoom_x), mode='reflect')

        # Swap z, y axes to convert to x-z view
        image_data = np.swapaxes(image_data, 0, 1)
        return image_data

    def apodize(self, image_data, napodize=10):
        (nz, ny, nx) = image_data.shape
        fact = 1 / napodize * math.pi * 0.5

        # Row-wise soften
        for y in range(0, napodize):
            # top
            image_data[:, y, :] = image_data[:, y, :] * math.pow(math.sin(fact * y), 2)
            # bottom
            image_data[:, ny - y - 1, :] = image_data[:, ny - y - 1, :] * math.pow(math.sin(fact * y), 2)

        # Column-wise soften
        for x in range(0, napodize):
            # left
            image_data[:, :, x] = image_data[:, :, x] * math.pow(math.sin(fact * x), 2)
            # right
            image_data[:, :, nx - x - 1] = image_data[:, :, nx - x - 1] * math.pow(math.sin(fact * x), 2)

        image_data[image_data < 0] = 0
        return image_data


if __name__ == "__main__":
    print('Training data (isotropic) generation for CARE training.\n')

    # region Folders for training data generation
    training_dir = r'X:\ForHari_stacksFromAndy\ER'
    input_folder = 'XZ_Input_50nm'
    truth_folder = 'XZ_GT_50nm'
    # endregion

    # Set the pixel size
    resolution = 0.05   # lateral pixel size of training images if not in file header, unit: um
    spacing = 0.125     # axial pixel size of training images if not in file header, unit: um
    use_header_flag = True  # Whether to use header of training images and ignore resolution & spacing above
    voxel_size = 0.05   # The final voxel size after scaling, unit: um

    # Set the blurring kernel (needs to be adjusted empirically)
    test_mode = True
    sigma_x = 2.6   # 2 ~ 3
    sigma_z = 1.3   # 0.5 ~ 1.5

    # Set resampling_flag: whether to mimic the coarse axial step size
    resampling_flag = True

    datagen_isotropic(training_dir=training_dir, input_folder=input_folder, truth_folder=truth_folder,
                      resolution=resolution, spacing=spacing, voxel_size=voxel_size, use_header_flag=use_header_flag,
                      test_mode=test_mode, sigma_x=sigma_x, sigma_z=sigma_z, resampling_flag=resampling_flag)
