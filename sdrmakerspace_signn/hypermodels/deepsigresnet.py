"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import tensorflow.python.keras.models as models
from tensorflow.python.keras.layers import (Input, Reshape, Dense,
                                            Flatten, BatchNormalization, Add,
                                            Activation, AlphaDropout)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          ZeroPadding2D,
                                                          MaxPooling2D)
from tensorflow.keras.regularizers import l2 as l2_reg

from tensorflow.python.keras.optimizers import Adam

from kerastuner import HyperModel


class deepsigresnet(HyperModel):

    def __init__(self, input_shape, target_num):
        self.input_shape = input_shape
        self.target_num = target_num

    def build(self, hp):
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
        
        nb_residual_units = hp.Int('residual_units',
                                   min_value=2,
                                   max_value=4,
                                   step=1)

        for i in range(hp.Int('residual_stacks',
                              min_value=4,
                              max_value=7,
                              step=1)):
            x = self._residual_stack(x, nb_channels=[32, 32, 32],
                                     nb_residual_units=nb_residual_units,
                                     name="resstack_" + str(i))

        x = Flatten()(x)

        dl_filters = hp.Choice('dense_filters',
                               [32, 64, 128])

        dr = hp.Float('dense_dr',
                      default=0.5,
                      min_value=0.3,
                      max_value=0.6,
                      step=0.1)

        for i in range(hp.Int('dense_layers',
                              min_value=2,
                              max_value=4,
                              step=1)):
            x = Dense(dl_filters,
                      activation='selu',
                      kernel_initializer='he_normal',
                      name="dense" + str(i))(x)
            x = AlphaDropout(dr)(x)

        x = Dense(self.target_num, name="output")(x)
        x = Activation('softmax')(x)
        x = Reshape([self.target_num])(x)
        model = models.Model(inp, x, name='deepsig_resnet')
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=Adam(hp.Choice(
                                        'learning_rate',
                                        [1e-2, 1e-3, 1e-4])))

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
                   kernel_regularizer=l2_reg(1.e-4),
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
                   kernel_regularizer=l2_reg(1.e-4),
                   name=name + "_conv_2")(x)
        # channels last
        x = BatchNormalization(axis=3, name=name + "_bn_2")(x)
        x = Add(name=name + "_add")([x, shortcut])
        x = Activation('relu', name=name + "_act_2")(x)
        return x

    def _residual_stack(self, inp, nb_channels, nb_residual_units, name):
        # 1x1 convolution
        x = Conv2D(nb_channels[0],
                   kernel_size=(1, 1),
                   padding='valid',
                   use_bias=False,
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2_reg(1.e-4),
                   name=name + "_conv_0")(inp)

        for i in range(nb_residual_units):
            x = self._residual_unit(x, nb_channels[1:3], [(1, 3), (1, 3)],
                                    name + "_resunit_" + str(i))

        x = MaxPooling2D(pool_size=(1, 2), name=name + "_mxpool_0")(x)
        return x
