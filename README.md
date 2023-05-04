# CARE

This repository is an updated and detailed version of [IsotropicZ_Github.py](https://github.com/eexuesong/SIMreconProject/tree/main/DeepLearning) in repository [SIMreconProject](https://github.com/eexuesong/SIMreconProject) to improve axial resolution in 3D SIM.

## Environment Configuration:
1. Install [Anaconda](https://www.anaconda.com/download) and [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows).

2. Create a conda environment.
    - In Pycharm, create a new project named e.g. "CARE" using [Conda environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html). A new environment with same name will also be created.
    - Or in [Anaconda Prompt](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), create a new environment named "CARE" by:
    ```posh
        conda create --name CARE
    ```
    To see a list of all your environments, type:
    ```posh
        conda info --envs
    ```
        or:
    ```posh
        conda env list
    ```
    Then in Pycharm, create a new project named e.g. "CARE" using Conda environment.
    Select the location of the Conda environment we just created as:
        C:\Users\username\AppData\Local\anaconda3\envs\CARE

3.  In Anaconda Prompt, activate the new environment:
        conda activate CARE

4.  GPU setup
    Important, or you will run DL on CPU instead. Without a GPU, training and prediction will be much slower
    (~30-60 times, even when using a computer with 40 CPU cores):
    First install NVIDIA GPU driver if you have not.

    For GPU support, it is very important to install the specific versions of CUDA and cuDNN
    that are compatible with the respective version of TensorFlow.
    Install the CUDA (11.2), cuDNN (8.1.0) https://www.tensorflow.org/install/source_windows#gpu:
        conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

5.  In Anaconda Promp, install TensorFlow 2.10:
        pip install tensorflow==2.10

6.  Verify the GPU setup:
        python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

7.  Finally, install the latest stable version of the CSBDeep package with pip.
    (https://pypi.org/project/csbdeep/#description)
    If you installed TensorFlow 2 (version 2.x.x):
        pip install csbdeep
    If you installed TensorFlow 1 (version 1.x.x):
        pip install "csbdeep[tf1]"
    Besides csbdeep, numpy, tifffile, scipy and matplotlib will also be installed meanwhile.
    Also install scikit-image for image transform:
        conda install -c conda-forge scikit-image

8.  Copy this file into the Pycharm project "CARE" folder and modify it accordingly.

9.1 In Pycharm, run the current file by Shift + F10
9.2 Or in Anaconda Prompt, run the Python file, e.g.:
        cd /D D:\Code\Python Code\CARE
        Python IsotropicZ_Github.py

10.  Use ctrl-C in the Terminal to terminate the process.

Note:
    (1) Do the following before initializing TensorFlow to limit TensorFlow to first GPU.
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    (2) You can find out which version of TensorFlow is installed via "pip show tensorflow".
