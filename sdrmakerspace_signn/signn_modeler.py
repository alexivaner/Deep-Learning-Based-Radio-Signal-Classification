"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import argparse
import tensorflow.python.keras.models as models
import tensorflow.keras.regularizers as regularizers
from tensorflow.python.keras.layers import (Reshape, Dense, Dropout,
                                            Activation, Flatten, Input,
                                            Add, BatchNormalization,
                                            AlphaDropout)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          ZeroPadding2D,
                                                          MaxPooling2D)


class signn_modeler():
    """
    A class that incorporates Tensorflow and Keras for the definition of the
    model architecture.

    Attributes
    ----------
    model : string
        the enumeration that defines the selected architecture
    input_shape : int
        a list of integers that defines the input shape of the architecture
    target_num : int
        the number of targets for the selected architecture
    destination: string
        the full path to save the Keras model containing the file name
        and extension
    """
    def __init__(self, model, input_shape, target_num, destination):
        self.model_choice = model
        self.input_shape = input_shape
        self.target_num = target_num
        self.model = self.__get_model()
        self.destination = destination

    def __get_model(self):
        if (self.model_choice == "deepsig"):
            return self.__get_deepsig_cnn()
        elif (self.model_choice == "deepsig_vgg"):
            return self.__get_deepsig_vgg()
        elif (self.model_choice == "deepsig_resnet"):
            return self.__get_deepsig_resnet()

    def __get_deepsig_cnn(self):
        """
        Return the model that describes the CNN architecture as described from
        Deepsig Inc.
        """
        dr = 0.5
        model = models.Sequential()
        model.add(Reshape(self.input_shape + [1],
                          input_shape=self.input_shape))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(256, (1, 3), activation="relu", name="conv1",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(80, (1, 3), activation="relu", name="conv12",
                         padding="valid", kernel_initializer="glorot_uniform"))
        model.add(Dropout(dr))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name="dense1"))
        model.add(Dropout(dr))
        model.add(Dense(self.target_num, name="dense2"))
        model.add(Activation('softmax'))
        model.add(Reshape([self.target_num]))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        return model

    def __get_deepsig_vgg(self, conv_layers=7, dense_layers=2, dr=0.5):
        """
        Return the model that describes the VGG CNN architecture as described
        from Deepsig Inc.
        """

        model = models.Sequential()
        model.add(Reshape(self.input_shape + [1],
                          input_shape=self.input_shape))
        model.add(ZeroPadding2D(padding=(0, 1)))
        model.add(Conv2D(filters=64,
                         kernel_size=(2, 3),
                         activation="relu",
                         name="conv0",
                         padding="valid",
                         kernel_initializer="glorot_uniform"))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(dr))

        # add convolutional layers
        for i in range(conv_layers-1):
            model.add(ZeroPadding2D(padding=(0, 1)))
            model.add(Conv2D(filters=64,
                             kernel_size=(1, 3),
                             activation="relu",
                             padding="valid",
                             kernel_initializer="glorot_uniform",
                             name="conv" + str(i+1)))
            model.add(MaxPooling2D(pool_size=(1, 2)))
            model.add(Dropout(dr))

        model.add(Flatten())

        # add dense layers
        for i in range(dense_layers):
            model.add(Dense(128,
                            activation='selu',
                            kernel_initializer="he_normal",
                            name="dense" + str(i)))
            model.add(AlphaDropout(dr))

        model.add(Dense(self.target_num, name="output"))
        model.add(Activation('softmax'))
        model.add(Reshape([self.target_num]))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        return model

    def _residual_unit(self, inp, nb_channels, kernel_sizes, name):
        shortcut = inp

        nb_channels1, nb_channels2 = nb_channels

        x = ZeroPadding2D(padding=(0, 1))(inp)
        x = Conv2D(nb_channels1,
                   kernel_sizes[0],
                   padding='valid',
                   use_bias=False,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(1.e-4),
                   name=name + "_conv_1")(x)
        # channels last
        x = BatchNormalization(axis=3, name=name + "_bn_1")(x)
        x = Activation('relu', name=name + "_act_1")(x)

        x = ZeroPadding2D(padding=(0, 1))(x)
        x = Conv2D(nb_channels2,
                   kernel_sizes[1],
                   padding='valid',
                   use_bias=False,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(1.e-4),
                   name=name + "_conv_2")(x)
        # channels last
        x = BatchNormalization(axis=3, name=name + "_bn_2")(x)
        x = Add(name=name + "_add")([x, shortcut])
        x = Activation('relu', name=name + "_act_2")(x)
        return x

    def _residual_stack(self, inp, nb_channels, kernel_sizes, name):
        # 1x1 convolution
        x = Conv2D(nb_channels[0],
                   kernel_size=(1, 1),
                   padding='valid',
                   use_bias=False,
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(1.e-4),
                   name=name + "_conv_0")(inp)

        x = self._residual_unit(x, nb_channels[1:3], kernel_sizes[0],
                                name + "_resunit0")
        x = self._residual_unit(x, nb_channels[1:3], kernel_sizes[1],
                                name + "_resunit1")

        x = MaxPooling2D(pool_size=(1, 2), name=name + "_mxpool_0")(x)
        return x

    def __get_deepsig_resnet(self, residual_stacks=6, dense_layers=2, dr=0.5):
        """
        Return the model that describes the ResNet architecture as described
        from Deepsig Inc.
        """

        inp = Input(shape=self.input_shape)

        x = Reshape(self.input_shape + [1])(inp)

        x = ZeroPadding2D(padding=(0, 1))(x)
        x = Conv2D(32,
                   kernel_size=(2, 3),
                   padding='valid',
                   kernel_initializer="glorot_uniform",
                   name="conv_0")(x)
        # channels last
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        for i in range(residual_stacks):
            x = self._residual_stack(x, nb_channels=[32, 32, 32],
                                     kernel_sizes=[
                                        [(1, 3), (1, 3)], [(1, 3), (1, 3)]],
                                     name="resstack_" + str(i))

        x = Flatten()(x)

        for i in range(dense_layers):
            x = Dense(128,
                      activation='selu',
                      kernel_initializer='he_normal',
                      name="dense" + str(i))(x)
            x = AlphaDropout(dr)(x)

        x = Dense(self.target_num, name="output")(x)
        x = Activation('softmax')(x)
        x = Reshape([self.target_num])(x)
        model = models.Model(inp, x, name='deepsig_resnet')
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        return model

    def export_model(self):
        """
        Save the generated architecture.
        """
        self.model.save(self.destination)

    def print_model_summary(self):
        """
        Print the summary of the generated architecture.
        """
        self.model.summary()


def argument_parser():
    description = 'A tool to generate models using Keras/Tensorflow'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-m", "--model", default='deepsig', nargs='?',
                        dest="model", action="store",
                        choices=['deepsig', 'deepsig_vgg', 'deepsig_resnet'],
                        help='Choose the model to generate. \
                            (Default: %(default)s)')
    parser.add_argument("-i", "--input-shape", dest="input_shape", nargs='+',
                        type=int, help='Set the model\'s input shape',
                        required=True)
    parser.add_argument("-n", "--target-num", dest="target_num", default=24,
                        type=int, help='Set the number of target classes.')
    parser.add_argument("-s", "--save", dest="destination", action='store',
                        help="Export the generated model at the given path.")
    parser.add_argument("-v", "--verbose", dest="verbose", action='store_true',
                        help="Print info regarding the generated model.")
    return parser


def main(modeler=signn_modeler, args=None):
    if args is None:
        args = argument_parser().parse_args()

    m = modeler(model=args.model, input_shape=args.input_shape,
                target_num=args.target_num, destination=args.destination,)
    if (args.destination):
        m.export_model()
    if (args.verbose):
        m.print_model_summary()


if __name__ == '__main__':
    main()
