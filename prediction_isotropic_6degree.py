from __future__ import print_function, unicode_literals, absolute_import, division
import time
import os
import numpy as np

import tifffile
import ImageJ_formatted_TIFF
from csbdeep.models import CARE
from scipy.ndimage import zoom, gaussian_filter1d, rotate
from skimage import transform

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit TensorFlow to first GPU.

# limit_gpu_memory(fraction=1/2)
r"""
The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
"""


class prediction_isotropic_6degree():
    def __init__(self, model_dir, prediction_dir, model_name, resolution=0.05, spacing=0.125, voxel_size=0.05,
                 use_header_flag=False, sigma_x=2.6, sigma_z=1.3, resampling_flag=True, save_flag=False):
        self.model_dir = model_dir
        self.prediction_dir = prediction_dir
        self.model_name = model_name
        self.rotation_angle = np.linspace(-90, 90, 7)

        # region Load trained model
        r"""
        Load trained model (located in base directory models with name model_name) from disk.
        The configuration was saved during training and is automatically loaded when CARE is initialized with config=None.
        """
        model = CARE(
            config=None,
            # csbdeep.models.Config or None – Valid configuration of CARE network.
            # If set to None, will be loaded from disk (must exist).
            name=self.model_name,
            # str or None – Model name. Uses a timestamp if set to None (default).
            basedir=self.model_dir
            # str – Directory that contains (or will contain) a folder with the given model name.
        )
        # endregion

        # region Create two folders for raw data (after scaling and swapping axes) and prediction output
        self.raw_dir = self.prediction_dir + '_raw'
        try:
            if not os.path.exists(self.raw_dir):
                os.makedirs(self.raw_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.raw_dir)
        else:
            print("Successfully created the directory %s " % self.raw_dir)

        self.output_dir = self.prediction_dir + '_CARE_6Degree_Isotropic'
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.output_dir)
        else:
            print("Successfully created the directory %s " % self.output_dir)
        # endregion

        self.voxel_size = voxel_size
        self.resampling_flag = resampling_flag
        self.save_flag = save_flag
        self.input_filenames = [f for f in os.listdir(self.prediction_dir) if f.endswith('.tif')]
        padding_pixel = 15
        axes = 'ZYX'

        # Apply CARE network to raw image
        for i in range(0, len(self.input_filenames)):
            print("processing... {0}".format(self.input_filenames[i]))

            input_filename = os.path.join(self.prediction_dir, self.input_filenames[i])
            (input_data, header) = ImageJ_formatted_TIFF.ReadTifStack(filename=input_filename)
            if not use_header_flag:
                header.resolution = resolution
                header.spacing = spacing

            (nz, ny, nx) = input_data.shape
            print("Image size [nz, ny, nx]: ({0}, {1}, {2})".format(nz, ny, nx))
            print('Image axes = {0}'.format(axes))

            # Scaling voxel size and swap z, y axes to get raw data
            raw_data = self.scale_voxel_swap_axes(data=input_data, lateral_pixel_size=header.resolution,
                                                  axial_pixel_size=header.spacing, voxel_size=voxel_size)
            # Write raw image data
            raw_filename = os.path.join(self.raw_dir, "raw_" + self.input_filenames[i])
            ImageJ_formatted_TIFF.WriteTifStack(numpy_array=raw_data, filename=raw_filename, resolution=self.voxel_size,
                                                spacing=self.voxel_size)
            (nz, ny, nx) = raw_data.shape
            print("Image size after scaling and swapping axes [nz, ny, nx]: ({0}, {1}, {2})".format(nz, ny, nx))

            # 1d Gaussian filter along current y-axis (previous z-axis) to suppress scaling artifacts in frequency domain.
            r"""
            Note by Xuesong:
                    We now use "order=3" in skimage.transform.resize which already suppresses high-frequency repetitions.
                    So sigma_z is unnecessary.
            """
            # input_data = gaussian_filter1d(raw_data, sigma_z, axis=1)
            input_data = raw_data

            # 1d Gaussian filter along x-axis to create a blurred but x-z isotropic input image data.
            input_data = gaussian_filter1d(input_data, sigma_x, axis=2)

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

            # Write blurred input
            if self.save_flag:
                blur_filename = os.path.join(self.output_dir, "blur_" + self.input_filenames[i])
                ImageJ_formatted_TIFF.WriteTifStack(numpy_array=input_data, filename=blur_filename,
                                                    resolution=self.voxel_size, spacing=self.voxel_size)

            results = np.zeros(shape=(len(self.rotation_angle) - 1, nz, ny, nx))  # (angle, nz, ny, nx)
            r"""
            Split the blurred input (x-z rectangle) into multiple tiles (x-z quasi-square)
            to reduce memory and graphics card requirements.
            """
            if nx >= ny:  # (nz, ny, nx) = (512, 100, 512)
                # In most cases, X dimension is larger than imaging depth
                tile_size = ny
                tile_number = nx / tile_size  # tile_number = 512 / 100 = 5.12
                overlap_size = max(10, int(nx / 50))
            else:
                tile_size = nx
                tile_number = ny / tile_size
                overlap_size = max(10, int(ny / 50))

            if tile_number == 1:
                # When nx == ny, only one tile
                tile_number = int(tile_number)
            else:
                # When nx != ny, more than one tile
                tile_number = int(tile_number) + 1  # tile_number = 6

            for tile_index in range(tile_number):  # tile_index = 0, 1, 2, 3, 4, 5
                print("number of tiles: {0} ...... processing tile #{1}".format(tile_number, tile_index + 1))
                start_pixel = tile_index * tile_size
                end_pixel = min((tile_index + 1) * tile_size, max(nx, ny))
                tile_pixel = end_pixel - start_pixel
                if tile_index == 0:
                    new_start_pixel = 0
                else:
                    new_start_pixel = overlap_size

                if nx >= ny:
                    tile_data = input_data[:, :, max(start_pixel - overlap_size, 0): min(end_pixel + overlap_size, nx)]
                else:
                    tile_data = input_data[:, max(start_pixel - overlap_size, 0): min(end_pixel + overlap_size, nx), :]
                r"""
                Suppose (nz, ny, nx) = (512, 100, 512) and overlap_size = 10
                tile_data = input_data[:, :, 0: 110]
                tile_data = input_data[:, :, 90: 210]
                tile_data = input_data[:, :, 190: 310]
                tile_data = input_data[:, :, 290: 410]
                tile_data = input_data[:, :, 390: 510]
                tile_data = input_data[:, :, 490: 512]
                """

                (nz_tile, ny_tile, nx_tile) = tile_data.shape
                print("tile data size [nz, ny, nx]: ({0}, {1}, {2})".format(nz_tile, ny_tile, nx_tile))
                if nx >= ny:
                    diagonal_pixel = max(int(np.sqrt(nx_tile * nx_tile + ny_tile * ny_tile)),
                                         int(ny_tile * np.sqrt(2)))
                else:
                    diagonal_pixel = max(int(np.sqrt(nx_tile * nx_tile + ny_tile * ny_tile)),
                                         int(nx_tile * np.sqrt(2)))

                pad_y = round((diagonal_pixel - ny_tile) / 2)
                pad_x = round((diagonal_pixel - nx_tile) / 2)
                pad_width = (0, 0), (pad_y, diagonal_pixel - ny_tile - pad_y), (pad_x, diagonal_pixel - nx_tile - pad_x)
                tile_data_pad = np.pad(array=tile_data, pad_width=pad_width, mode="reflect")
                [nz_tile_pad, ny_tile_pad, nx_tile_pad] = tile_data_pad.shape
                print("after padding [nz, ny, nx]: ({0}, {1}, {2})".format(nz_tile_pad, ny_tile_pad, nx_tile_pad))

                # Rotation
                for angle in range(len(self.rotation_angle) - 1):
                    if self.rotation_angle[angle] == 0:
                        # Special case #1: use "tile_data" directly, no rotation
                        tile_data_restored = model.predict(img=tile_data, axes=axes)
                    elif self.rotation_angle[angle] == -90:
                        # Special case #2: rotate "tile_data" and then predict
                        input_tile = np.rot90(m=tile_data, k=1, axes=(1, 2))
                        output_tile = model.predict(img=input_tile, axes=axes)
                        tile_data_restored = np.rot90(m=output_tile, k=1, axes=(2, 1))
                    else:
                        # Other cases: use "tile_data_pad" instead of "tile_data"
                        input_tile = rotate(input=tile_data_pad, angle=-self.rotation_angle[angle], axes=(1, 2),
                                            reshape=False)
                        output_tile = model.predict(img=input_tile, axes=axes, n_tiles=(1, 1, 1))
                        output_tile = rotate(input=output_tile, angle=self.rotation_angle[angle], axes=(1, 2),
                                             reshape=False)
                        tile_data_restored = output_tile[:, pad_y:pad_y + ny_tile, pad_x:pad_x + nx_tile]

                    [nz_tile_restored, ny_tile_restored, nx_tile_restored] = tile_data_restored.shape
                    print("prediction: {0:d} degrees [nz, ny, nx]: ({1}, {2}, {3})".format(
                        int(self.rotation_angle[angle]),
                        nz_tile_restored,
                        ny_tile_restored,
                        nx_tile_restored))

                    if nx >= ny:
                        results[angle, :, :, start_pixel:end_pixel] = tile_data_restored[:, :,
                                                                  new_start_pixel:new_start_pixel + tile_pixel]
                    else:
                        results[angle, :, start_pixel:end_pixel, :] = tile_data_restored[:,
                                                                  new_start_pixel:new_start_pixel + tile_pixel, :]
                    r"""
                    Suppose (nz, ny, nx) = (512, 100, 512) and overlap_size = 10
                    results[k, :, :, 0:100] = tile_data_restored[:, :, 0: 100]
                    results[k, :, :, 100:200] = tile_data_restored[:, :, 10: 110]
                    ...
                    results[k, :, :, 500:512] = tile_data_restored[:, :, 10: 22] 
                    """

            results[results <= 0] = 0
            results = results[:, :, padding_pixel:ny - padding_pixel, padding_pixel:nx - padding_pixel]
            if self.save_flag:
                for angle in range(len(self.rotation_angle) - 1):
                    angle_filename = os.path.join(self.output_dir, "DL_" + str(int(self.rotation_angle[angle])) + "_" +
                                                  self.input_filenames[i])
                    ImageJ_formatted_TIFF.WriteTifStack(numpy_array=results[angle, :, :, :].astype("single"),
                                                        filename=angle_filename, resolution=self.voxel_size,
                                                        spacing=self.voxel_size)
            print("DL results shape: {0}".format(results.shape))

            # region Combine 7-angle results in frequency domain (take the biggest value from 7 angles)
            mean_angle = np.mean(a=results, axis=(1, 2, 3))
            mean_max = np.amax(a=mean_angle)
            results_fft = np.zeros_like(a=results, dtype=complex)
            for angle in range(len(self.rotation_angle) - 1):
                # Average the 7-angle images so they have the same total intensity.
                results[angle, :, :, :] = results[angle, :, :, :] * mean_max / mean_angle[angle]
                results_fft[angle, :, :, :] = np.fft.fftn(a=results[angle, :, :, :])
            results_fft_abs = np.absolute(results_fft)

            max_index = np.argmax(a=results_fft_abs, axis=0)
            results_fft_max = np.zeros_like(a=raw_data, dtype=complex)

            for angle in range(len(self.rotation_angle) - 1):
                fft_abs_mask = np.zeros_like(a=raw_data)
                fft_abs_mask[max_index == angle] = 1
                # Take the maximum fft data based on their absolute values
                results_fft_max = results_fft_max + results_fft[angle, :, :, :] * fft_abs_mask

            results_fft_max = np.fft.ifftn(a=results_fft_max)
            results_fft_max_abs = np.absolute(results_fft_max).astype("single")

            # Write output image data
            output_filename = os.path.join(self.output_dir, "max_fft_abs_" + self.input_filenames[i])
            ImageJ_formatted_TIFF.WriteTifStack(numpy_array=results_fft_max_abs, filename=output_filename,
                                                resolution=self.voxel_size, spacing=self.voxel_size)

            if self.save_flag:  # For comparison between abs and real.
                real_filename = os.path.join(self.output_dir, "max_fft_real_" + self.input_filenames[i])
                ImageJ_formatted_TIFF.WriteTifStack(numpy_array=np.real(val=results_fft_max).astype("single"),
                                                    filename=real_filename, resolution=self.voxel_size,
                                                    spacing=self.voxel_size)
            print()
            # endregion

    def scale_voxel_swap_axes(self, data, lateral_pixel_size=0.05, axial_pixel_size=0.125, voxel_size=0.05):
        # Change the image voxel size to be the same (e.g. 50 nm) in x, y, z
        (nz, ny, nx) = data.shape
        nz_new = round(nz * axial_pixel_size / voxel_size)
        ny_new = round(ny * lateral_pixel_size / voxel_size)
        nx_new = round(nx * lateral_pixel_size / voxel_size)
        data = transform.resize(image=data, output_shape=[nz_new, ny_new, nx_new], order=3, mode="reflect",
                                preserve_range=True)
        r"""
        Note by Xuesong: if using "order=3" in skimage.transform.resize, sigma_z is unnecessary.
        """

        # Swap z, y axes to convert to x-z view
        data = np.swapaxes(data, 0, 1)
        return data


if __name__ == "__main__":
    print('CARE prediction (Isotropic 6 degree).\n')

    # region Folders for prediction
    model_dir = r'X:\shrofflab\Xuesong\Xuesong_data\3D_SIM_DL\2022_02_02_001_Jurkat_EMTB_EGFP_high_low_SNR(Nikon)\Wiener_reconstruction(highSNR)\50nm'
    model_name = 'CARE_XZ_6degree_Model_EMTB_50nm'

    prediction_dir = r'X:\ForHari_stacksFromAndy\Vimentin_ER_Zstacks\Movies\2021-03-02_C2_Z39_T400\ER'
    # endregion

    # Set the pixel size
    resolution = 0.05  # lateral pixel size of input images if not in file header, unit: um
    spacing = 0.125  # axial pixel size of input images if not in file header, unit: um
    use_header_flag = True  # Whether to use header of input images and ignore resolution & spacing above
    voxel_size = 0.05  # The final voxel size after scaling, unit: um

    # Set the blurring kernel (needs to be consistent with the settings in "datagen_isotropic.py")
    sigma_x = 2.6  # 2 ~ 3
    sigma_z = 1.3  # 0.5 ~ 1.5

    # Set resampling_flag: whether to mimic the coarse axial step size
    resampling_flag = True

    # Set save_flag: whether to save intermediate files
    save_flag = True

    prediction_isotropic_6degree(model_dir=model_dir, prediction_dir=prediction_dir, model_name=model_name,
                                 resolution=resolution, spacing=spacing, voxel_size=voxel_size,
                                 use_header_flag=use_header_flag, sigma_x=sigma_x,
                                 sigma_z=sigma_z, resampling_flag=resampling_flag, save_flag=save_flag)
