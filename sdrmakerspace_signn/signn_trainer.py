"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import argparse
import os
import errno
import time
import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow.python.keras.models as models
import tensorflow.python.keras.callbacks as clbck
from datetime import datetime

from utils import dataset_generator as dg
from utils import plotter as plt


class signn_trainer():
    """
    A class that incorporates Tensorflow and Keras for the training process.

    Attributes
    ----------
    dataset_path : string
        the path to the directory of the dataset
    dataset_name : string
        the name including the extension of the model file
    model_path : string
        the full path of the Keras model containing the file name and extension
    epochs : int
        the total number of full training passes over the entire dataset
    steps_per_epoch : int
        the total number of steps (batches of samples) before declaring one
        epoch finished
    batch_size : int
        the number of samples per gradient update
    shuffle : boolean
        a switch to enable/disable the shuffling of dataset
    shuffle_buffer_size : int
        the size of buffer that will be filled for the shuffling of dataset
    split_ratio : float
        a triplet of floats that define the train/validation/test portions of
        the dataset
    dataset_shape : int
        a vector of ints that define the shape of the dataset
    validation_steps : int
        the total number of steps (batches of samples) to draw before stopping
        when performing validation at the end of every epoch
    artifacts_dest : str
        the path to the directory of training artifacts
    dataset_percent : float
        percentage of total dataset to use (subset of each SNR and modulation
        selected)
    data_transform : str
        type of transformation to apply to dataset examples 
    snr: int
        list of integers that defines a subset of samples with specific SNR
        from the selected dataset
    modulation: string
        list of strings that defines a subset of samples with specific
        modulation samples from the selected dataset
    enable_gpu : boolean
        a switch to enable/disable the GPU accelerator
    """
    def __init__(self, dataset_path, dataset_name, model_path, epochs,
                 steps_per_epoch, batch_size, shuffle, shuffle_buffer_size,
                 split_ratio, dataset_shape, validation_steps, artifacts_dest,
                 dataset_percent, data_transform, snr, modulation, enable_gpu):
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.split_ratio = split_ratio.copy()
        self.dataset_shape = dataset_shape.copy()
        self.validation_steps = validation_steps
        self.enable_gpu = enable_gpu
        self.__configure_accelerators()
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.dataset_parser = dg.dataset_generator(
            self.dataset_path,
            dataset_name,
            dataset_percent=dataset_percent,
            data_transform=data_transform,
            snr=snr,
            modulation=modulation,
            split_ratio=split_ratio)
        self.train_samples_num = int((self.split_ratio[0]*self.dataset_parser
                                      .get_total_samples()))
        self.validation_samples_num = int((self.split_ratio[1] *
                                           self.dataset_parser
                                           .get_total_samples()))
        self.test_samples_num = int((self.split_ratio[2] *
                                     self.dataset_parser.get_total_samples()))
        self.__init_dataset()
        self.__init_model()
        self.logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.artifacts_dest = artifacts_dest
        self.file_writer_cm = tf.summary.create_file_writer(
            self.logdir + '/cm')
        self.plotter = plt.plotter(artifacts_dest)

    def __configure_accelerators(self):
        if self.enable_gpu is not True:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Ask TensorFlow to only allocate 6GB of memory on first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=6000)])
                    logical_gpus = tf.config.experimental.list_logical_devices(
                        'GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus),
                          "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs
                    # have been initialized
                    print(e)

    def __init_dataset(self):
        """
        A method to initialize the train, validation and test datasets by\
            by calling the appropriate generators respectively. 
        """
        if (not os.path.exists(self.dataset_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.dataset_path)
        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.train_dataset_generator(),
            (tf.float32, tf.uint8), (self.dataset_shape, []))
        print("Train dataset initialization done.")
        self.validation_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.validation_dataset_generator(),
            (tf.float32, tf.uint8), (self.dataset_shape, []))
        print("Validation dataset initialization done.")
        self.test_dataset = tf.data.Dataset.from_generator(
            lambda: self.dataset_parser.test_dataset_generator(),
            (tf.float32, tf.uint8), (self.dataset_shape, []))
        print("Test dataset initialization done.")
        if self.shuffle:
            self.train_dataset = self.train_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=int(round(time.time() * 1000)),
                reshuffle_each_iteration=False)
            print("Shuffling datasets on.")
        self.train_dataset = self.train_dataset.batch(
            batch_size=self.batch_size)
        self.validation_dataset = self.validation_dataset.batch(
            batch_size=self.batch_size)
        self.test_dataset = self.test_dataset.batch(
            batch_size=self.batch_size)
        print("Batch sizes set on datasets.")

    def __init_model(self):
        """
        A method to load the Keras model from the predefined path.
        """
        if (not os.path.isfile(self.model_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.model_path)
        self.model = models.load_model(self.model_path)

    def predict(self):
        """
        A method that uses the trained model to make predictions over the test\
            dataset.
        """
        predictions = self.model.predict(self.test_dataset)
        truth_labels = np.array([])
        for i in self.test_dataset:
            truth_labels = np.concatenate((truth_labels, np.array(i[1])),
                                          axis=None)
        cm = sklearn.metrics.confusion_matrix(truth_labels,
                                              np.argmax(predictions, axis=1))
        # TODO: Handle the case of modulation subset
        # cm = cm[len(np.unique(truth_labels)):,
        #         0:len(np.unique(truth_labels))]
        self.plotter.plot_confusion_matrix(
            cm, self.dataset_parser.list_available_modulations(),
            "conf_new.png")
        # self.plotter.plot_training_validation_loss()

    def __log_confusion_matrix(self):
        """
        A method to generate and log the confussion matrix from the test\
            dataset predictions.
        """
        print("Logging to Tensorboard")
        predictions = self.model.predict(self.test_dataset)
        truth_labels = np.array([])
        for i in self.test_dataset:
            truth_labels = np.concatenate((truth_labels, np.array(i[1])),
                                          axis=None)
        cm = sklearn.metrics.confusion_matrix(truth_labels,
                                              np.argmax(predictions, axis=1))
        # TODO: Handle the case of modulation subset
        # cm = cm[len(np.unique(truth_labels)):,
        #         0:len(np.unique(truth_labels))]
        figure = self.plotter.plot_confusion_matrix(
            cm, self.dataset_parser.list_selected_modulations(),
            "conf_new.png")
        cm_image = self.plotter.plot_to_image(figure)

        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=1)

    def train(self):
        """
        A method that trains and saves a Keras model. 
        """
        filepath = self.artifacts_dest+"/trained_model.h5"
        callback_list = [
            clbck.ModelCheckpoint(filepath, monitor='val_loss', verbose=2,
                                  save_best_only=True, mode='auto'),
            clbck.EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                mode='auto'),
            clbck.TensorBoard(log_dir=self.logdir),
            clbck.LambdaCallback(on_epoch_end=lambda epoch, logs:
                                 (self.__log_confusion_matrix()))]

        self.history = self.model.fit(x=self.train_dataset,
                                      epochs=self.epochs,
                                      steps_per_epoch=self.steps_per_epoch,
                                      validation_data=self.validation_dataset,
                                      validation_steps=self.validation_steps,
                                      verbose=2,
                                      shuffle=False,
                                      workers=16,
                                      use_multiprocessing=True,
                                      callbacks=callback_list)
        self.model.load_weights(filepath)


def argument_parser():
    description = 'A tool to train a CNN using Keras/Tensorflow'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-p --dataset-path", dest="dataset_path",
                        required=True, action="store",
                        help="Set dataset path.")
    parser.add_argument("-d --dataset-name", dest="dataset_name",
                        required=True, action="store",
                        help="Set full dataset name.")
    parser.add_argument("-m --model-path", dest="model_path", action="store",
                        default="", help="Set model path.")
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=100, help="Set batch size.")
    parser.add_argument("--epochs", dest="epochs", type=int, default=30,
                        help="Set training epochs.")
    parser.add_argument("--steps-per-epoch", dest="steps_per_epoch", type=int,
                        default=None, help="Set training steps per epoch.")
    parser.add_argument('--shuffle', dest="shuffle", action='store_true',
                        help="Shuffle the dataset.")
    parser.add_argument('--no-shuffle', dest="shuffle", action='store_false',
                        help="Do not shuffle the dataset.")
    parser.add_argument("--shuffle-buffer-size", dest="shuffle_buffer_size",
                        type=int, default=10000,
                        help="Set shuffle buffer size.")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--split-ratio", default=[0.8, 0.1, 0.1], nargs='+',
                        dest="split_ratio", action="store", type=float,
                        help='Set the train/validation portions. \
                            (Default: %(default)s)')
    parser.add_argument("--dataset-shape", default=[2, 1024], nargs='+',
                        dest="dataset_shape", action="store", type=int,
                        help='Set the dataset shape. \
                            (Default: %(default)s)')
    parser.add_argument("--validation-steps", dest="validation_steps",
                        type=int, default=None,
                        help="Set the number of validation steps.")
    parser.add_argument("--artifacts-directory", dest="artifacts_dest",
                        default="artifacts",
                        help="Set the destination folder of the training\
                            artifacts.")
    parser.add_argument('--train', dest="train", action='store_true',
                        help="Enable training.")
    parser.add_argument('--no-train', dest="train", action='store_false',
                        help="Disable training.")
    parser.set_defaults(train=False)
    parser.add_argument('--test', dest="test", action='store_true',
                        help="Enable testing.")
    parser.add_argument('--no-test', dest="test", action='store_false',
                        help="Disable testing.")
    parser.set_defaults(test=False)
    parser.add_argument("--dataset-percent", dest='dataset_percent',
                        type=float, default=1,
                        help="Define percentage of original dataset to use")
    parser.add_argument("--data-transform", dest='data_transform',
                        default='cartesian',
                        help="Apply transform to dataset examples.")
    parser.add_argument("--snr", default="-20", nargs='+',
                        dest="snr", action="store", type=int,
                        help='Set the SNR samples to extract from dataset. \
                            (Default: %(default)s)')
    parser.add_argument("--modulation", default=None, nargs='+',
                        dest="modulation", action="store",
                        help='Set the modulation samples to extract from dataset. \
                            (Default: %(default)s)')
    parser.add_argument('--gpu', dest="enable_gpu", action='store_true',
                        help="Enable GPU.")
    parser.add_argument('--cpu', dest="enable_gpu", action='store_false',
                        help="Use CPU only.")
    parser.set_defaults(enable_gpu=True)
    return parser


def main(trainer=signn_trainer, args=None):
    if args is None:
        args = argument_parser().parse_args()

    t = trainer(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                model_path=args.model_path, epochs=args.epochs,
                steps_per_epoch=None, batch_size=args.batch_size,
                shuffle=args.shuffle,
                shuffle_buffer_size=args.shuffle_buffer_size,
                split_ratio=args.split_ratio,
                dataset_shape=args.dataset_shape,
                validation_steps=None,
                artifacts_dest=args.artifacts_dest,
                dataset_percent=args.dataset_percent,
                data_transform=args.data_transform,
                snr=[args.snr],
                modulation=args.modulation,
                enable_gpu=args.enable_gpu)

    if args.train:
        t.train()
    if args.test:
        t.predict()


if __name__ == '__main__':
    main()
