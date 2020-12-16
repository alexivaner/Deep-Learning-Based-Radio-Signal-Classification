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
import glob
import numpy as np
import sklearn.metrics
import tensorflow as tf
from datetime import datetime

from utils import dataset_generator as dg
from utils import plotter as plt

from kerastuner.tuners import (RandomSearch, BayesianOptimization)
import tensorflow.python.keras.models as models

import hypermodels


class signn_tuner():
    """
    A class that incorporates Tensorflow and Keras for the hyperparameter
    tuning of NN models.

    Attributes
    ----------
    dataset_path : string
        the path to the directory of the dataset
    dataset_name : string
        the name including the extension of the model file
    target_num : int
        the number of target classes
    model : string
        selected NN architecture
    epochs : int
        the total number of full training passes over the entire dataset
    batch_size : int
        the number of samples per gradient update
    shuffle : boolean
        a switch to enable/disable the shuffling of dataset
    shuffle_buffer_size : int
        the size of buffer that will be filled for the shuffling of dataset
    tuner_type : string
        selected hyperparameter tuner type
    max_trials : int
        the maximum number of tuning trials
    best_models_num : int
        the maximum number of top best models to use for prediction on test set
    destination : string
        the full path to save the tuning results
    split_ratio : float
        a triplet of floats that define the train/validation/test portions of
        the dataset
    dataset_shape : int
        a vector of ints that define the shape of the dataset
    dataset_percent : float
        percentage of total dataset to use (subset of each SNR and modulation
        selected)
    data_transform : str
        type of transformation to apply to dataset examples
    do_testing : bolean
        a switch to enable/disable testing on top best models 
    snr: int
        list of integers that defines a subset of samples with specific SNR
        from the selected dataset
    modulation: string
        list of strings that defines a subset of samples with specific
        modulation samples from the selected dataset
    enable_gpu : boolean
        a switch to enable/disable the GPU accelerator
    """

    def __init__(self, dataset_path, dataset_name,
                 target_num, model, epochs,
                 batch_size, shuffle, shuffle_buffer_size,
                 tuner_type, max_trials,
                 best_models_num, destination,
                 split_ratio, dataset_shape,
                 dataset_percent, data_transform,
                 do_testing,
                 snr, modulation, enable_gpu):
        self.target_num = target_num
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.tuner_type = tuner_type
        self.max_trials = max_trials
        self.best_models_num = best_models_num
        self.destination = os.path.join(destination,
                                        datetime.now().
                                        strftime("%Y%m%d-%H%M%S"))
        self.split_ratio = split_ratio.copy()
        self.dataset_shape = dataset_shape.copy()
        self.do_testing = do_testing
        self.enable_gpu = enable_gpu
        self.__configure_accelerators()
        self.dataset_path = dataset_path
        self.model = model
        self.dataset_parser = dg.dataset_generator(
            self.dataset_path,
            dataset_name,
            dataset_percent=dataset_percent,
            data_transform=data_transform,
            snr=snr,
            modulation=modulation,
            split_ratio=split_ratio)

        self.__init_dataset()
        self.__init_model()
        self.__init_tuner()

        if do_testing:
            self.__init_plotter()

    def __configure_accelerators(self):
        """
        A method to configure GPU accelerators.
        """
        if self.enable_gpu is not True:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
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
                    print(e)

    def __init_dataset(self):
        """
        A method to initialize the train and validation datasets
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
        A method to initialize the appropriate hypermodel.
        """
        if (self.model == "deepsig"):
            self.hypermodel = hypermodels.deepsigcnn(self.dataset_shape,
                                                     self.target_num)
        elif (self.model == "deepsigvgg"):
            self.hypermodel = hypermodels.deepsigvgg(self.dataset_shape,
                                                     self.target_num)
        elif (self.model == "deepsigresnet"):
            self.hypermodel = hypermodels.deepsigresnet(self.dataset_shape,
                                                        self.target_num)

    def __init_tuner(self):
        """
        A method to initialize the selected hyperparameter tuner.
        """
        if (self.tuner_type == "randomsearch"):
            self.tuner = RandomSearch(self.hypermodel,
                                      objective='val_accuracy',
                                      max_trials=self.max_trials,
                                      directory=self.destination,
                                      project_name='signn_tuning')
        elif (self.tuner_type == "bayesian"):
            self.tuner = BayesianOptimization(self.hypermodel,
                                              objective='val_accuracy',
                                              max_trials=self.max_trials,
                                              directory=self.destination,
                                              project_name='signn_tuning')

    def __init_plotter(self):
        """
        A method to initialize confusion matrix plotter.
        """
        pred_cm_path = os.path.join(self.destination, 'prediction_cm')
        if (not os.path.exists(pred_cm_path)):
            os.mkdir(pred_cm_path)
        self.plotter = plt.plotter(os.path.join(self.destination,
                                                'prediction_cm'))

    def predict(self):
        """
        A method that uses at max [best_models_num] models to make predictions
        over the test dataset.
        """
        # load model from file in order to avoid issue in
        # https://github.com/tensorflow/tensorflow/issues/33997
        
        best_models_path = os.path.join(self.destination, 'best_models/*.h5')
        for idx, filename in enumerate(glob.glob(best_models_path)):
            model = models.load_model(filename)
            predictions = model.predict(self.test_dataset)
            true_labels = np.array([])
            for i in self.test_dataset:
                true_labels = np.concatenate((true_labels, np.array(i[1])),
                                             axis=None)
            predicted_labels = np.argmax(predictions, axis=1)

            print("Best model {0} test accuracy: {1:.4f}"
                  .format(idx,
                          sklearn.metrics.accuracy_score(true_labels,
                                                         predicted_labels)))

            cm = sklearn.metrics.confusion_matrix(true_labels,
                                                  predicted_labels)
            self.plotter.plot_confusion_matrix(
                cm, self.dataset_parser.list_available_modulations(),
                "best_model_" + str(idx) + ".jpg")

    def save_best_models(self):
        """
        A method that saves at max [best_models_num] models.
        """
        best_models_path = os.path.join(self.destination, 'best_models')
        if (not os.path.exists(best_models_path)):
            os.mkdir(best_models_path)
        best_models = self.tuner.get_best_models(self.best_models_num)
        for model_idx, model in enumerate(best_models):
            model.save(os.path.join(best_models_path,
                                    'best_model_' + str(model_idx) + '.h5'))

    def tune(self):
        """
        A method that tunes the hyperparameters of the Keras model.
        """
        self.tuner.search(x=self.train_dataset,
                          epochs=self.epochs,
                          validation_data=self.validation_dataset,
                          workers=16,
                          shuffle=False,
                          use_multiprocessing=True)

        self.tuner.results_summary()


def argument_parser():
    description = 'A tool to tune the hyperparameters of a Keras model'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-p --dataset-path", dest="dataset_path",
                        required=True, action="store",
                        help="Set dataset path.")
    parser.add_argument("-d --dataset-name", dest="dataset_name",
                        required=True, action="store",
                        help="Set full dataset name.")
    parser.add_argument("-m --model", dest="model", action="store",
                        choices=['deepsig', 'deepsigvgg', 'deepsigresnet'],
                        default="deepsig",
                        help="Choose the model to tune (Default: %(default)s)")
    parser.add_argument("-n --target-num", dest="target_num", default=24,
                        type=int, help='Set the number of target classes.')
    parser.add_argument("-s --save", dest="destination",
                        action='store', required=True,
                        help="Specify path for saving tuning results.")
    parser.add_argument("--tuner-type", dest="tuner_type",
                        action="store",
                        choices=['randomsearch', 'bayesian'],
                        default="randomsearch",
                        help="Set tuner type (Default: %(default)s)")
    parser.add_argument("--max-trials", dest="max_trials", type=int,
                        default=10, help="Set max tuning trials")
    parser.add_argument("--best-models", dest="best_models_num", type=int,
                        default=1, help="Set top best models to save and use\
                        for prediction")
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=100, help="Set batch size.")
    parser.add_argument("--epochs", dest="epochs", type=int, default=30,
                        help="Set training epochs.")
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
                        help='Set the train/validation/test portions. \
                            (Default: %(default)s)')
    parser.add_argument('--test', dest="do_testing", action='store_true',
                        help="Enable testing.")
    parser.add_argument('--no-test', dest="do_testing", action='store_false',
                        help="Disable testing.")
    parser.set_defaults(do_testing=False)
    parser.add_argument("--dataset-shape", default=[2, 1024], nargs='+',
                        dest="dataset_shape", action="store", type=int,
                        help='Set the dataset shape. \
                            (Default: %(default)s)')
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


def main(tuner=signn_tuner, args=None):
    if args is None:
        args = argument_parser().parse_args()

    t = tuner(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
              target_num=args.target_num, model=args.model,
              epochs=args.epochs, batch_size=args.batch_size,
              shuffle=args.shuffle,
              shuffle_buffer_size=args.shuffle_buffer_size,
              tuner_type=args.tuner_type, max_trials=args.max_trials,
              best_models_num=args.best_models_num,
              destination=args.destination, do_testing=args.do_testing,
              split_ratio=args.split_ratio, dataset_shape=args.dataset_shape,
              dataset_percent=args.dataset_percent,
              data_transform=args.data_transform,
              snr=[args.snr],
              modulation=args.modulation,
              enable_gpu=args.enable_gpu)

    t.tune()
    t.save_best_models()

    if args.do_testing:
        t.predict()


if __name__ == '__main__':
    main()
