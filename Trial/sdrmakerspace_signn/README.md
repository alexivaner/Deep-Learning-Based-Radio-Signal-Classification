## signn: Signal Detection and Classification using Neural Networks


## Requirements
The requirements are listed in the *requirements.txt* located in the root directory of this project.

* Tensorflow [2.0.0]
* Keras [2.2.4]
* sklearn
* numpy [1.16.2]
* matplotlib [3.1.0]
* h5py [2.9.0]
* keras-tuner [1.0.0]

In addition, for running some auxilliary scripts some extra dependencies should be met:

*  GNURadio [3.7+]

## Download & Install

~~~~
$ git clone https://gitlab.com/librespacefoundation/sdrmakerspace/signn.git
$ cd signn
$ sudo pip install -r requirements.txt
~~~~

In order to install GNURadio 3.7 please check for the appropriate developement package for your distribution.

## Usage

##### 1. Generate the signal dataset:

First download and extract the source material need for the generation of signals:

~~~~
$ cd utils/dataset/gnuradio_sim
$ wget https://cloud.libre.space/s/rzS3QaXLY6BTN3x/download
$ tar xvzf source_material.tar.gz
~~~~

Then, run the dataset generation script:
~~~~
$ python2 generate_sim_dataset.py
~~~~

**Note:** The `generate_sim_dataset.py` requires python2.7 as it depends on GNURadio 3.7+.

This command will create the *SIGNN_2019_01.hdf5* dataset. This step can be skipped in order to use a different signal dataset, as long as it follows the specific scheme described here.

##### 2. Generate the Keras model used for training:

~~~~
$ python3 signn_modeler.py -i 2 1024 -s model.h5
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_modeler.py --help
~~~~

##### 3. Train the Keras model exported with the previous command:

~~~~
$ python3 signn_trainer.py -p utils/dataset/gnuradio_sim -d SIGNN_2019_01.hdf5 -m model.h5 --train
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_trainer.py --help
~~~~

##### 4. Monitor the training progress using Tensorboard:

~~~~
$ tensorboard --logdir artifacts/plots
~~~~

Then navigate to http://0.0.0.0:6006


##### 5. Tune Keras model hyperparameters by using keras-tuner library.

~~~~
$ python3 signn_tuner.py -p utils/dataset/gnuradio_sim -d SIGNN_2019_01.hdf5 --dataset-shape 2 1024
~~~~

For more information, please use the help argument:

~~~~
$ python3 signn_tuner.py --help
~~~~
