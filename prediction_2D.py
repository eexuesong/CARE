from __future__ import print_function, unicode_literals, absolute_import, division
import time
import os
import numpy as np

import tifffile
import ImageJ_formatted_TIFF
from csbdeep.models import CARE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit TensorFlow to first GPU.

# limit_gpu_memory(fraction=1/2)
r"""
The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
"""

class prediction_2d():
    def __init__(self, model_dir, prediction_dir, model_name, resolution=0.05, use_header_flag=False):
        self.model_dir = model_dir
        self.prediction_dir = prediction_dir
        self.model_name = model_name

        self.output_dir = prediction_dir + '_CARE_2D'
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.output_dir)
        else:
            print("Successfully created the directory %s " % self.output_dir)

        self.input_filenames = [f for f in os.listdir(self.prediction_dir) if f.endswith('.tif')]
        axes = 'YX'

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

        # region Apply CARE network to raw image
        for i in range(0, len(self.input_filenames)):
            print("processing... {0}".format(self.input_filenames[i]))
            input_filename = os.path.join(self.prediction_dir, self.input_filenames[i])
            (input_data, header) = ImageJ_formatted_TIFF.ReadTifStack(filename=input_filename)
            if not use_header_flag:
                header.resolution = resolution
            print('image size = {0}'.format(input_data.shape))
            print('image axes = {0}'.format(axes))
            restored_data = model.predict(img=input_data, axes=axes)
            # Save restored image
            output_filename = os.path.join(self.output_dir, 'DL_' + self.input_filenames[i])
            ImageJ_formatted_TIFF.WriteTifStack(numpy_array=restored_data, filename=output_filename,
                                                resolution=header.resolution)
            print()
        # endregion


if __name__ == "__main__":
    print('CARE prediction.\n')

    # region Folders for prediction
    model_dir = r''
    model_name = ''

    prediction_dir = r''
    # endregion

    # Set the pixel size
    resolution = 0.04088  # lateral pixel size of input images if not in file header, unit: um
    use_header_flag = False  # Whether to use header of input images and ignore resolution above

    prediction_2d(model_dir=model_dir, prediction_dir=prediction_dir, model_name=model_name, resolution=resolution,
                  use_header_flag=use_header_flag)
