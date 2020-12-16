"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import tensorflow.python.keras.models as models
from tensorflow.python.keras.layers import (Reshape, Dense,
                                            Flatten, BatchNormalization,
                                            Activation, AlphaDropout)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          ZeroPadding2D,
                                                          MaxPooling2D)
from tensorflow.python.keras.optimizers import Adam

from kerastuner import HyperModel


class deepsigvgg(HyperModel):

    def __init__(self, input_shape, target_num):
        self.input_shape = input_shape
        self.target_num = target_num

    def build(self, hp):       
        model = models.Sequential()
        model.add(Reshape(self.input_shape + [1],
                          input_shape=self.input_shape))
        model.add(ZeroPadding2D(padding=(0, 1)))
        model.add(Conv2D(filters=hp.Int('conv1_filters',
                                        min_value=16,
                                        max_value=64,
                                        step=16),
                         kernel_size=(1, 3),
                         name="conv0",
                         padding="valid",
                         kernel_initializer="glorot_uniform"))
        model.add(BatchNormalization(axis=3, name='bn0'))
        model.add(Activation('relu', name="relu0"))
        model.add(MaxPooling2D(pool_size=(1, 2)))

        # add convolutional layers
        for i in range(hp.Int('conv_layers', 3, 6)):
            model.add(ZeroPadding2D(padding=(0, 1)))
            model.add(Conv2D(filters=hp.Int('conv' + str(i+1) + '_filters',
                                            min_value=16,
                                            max_value=64,
                                            step=16),
                             kernel_size=(1, 3),
                             padding="valid",
                             kernel_initializer="glorot_uniform",
                             name="conv" + str(i+1)))
            model.add(BatchNormalization(axis=3, name='bn' + str(i+1)))
            model.add(Activation('relu', name="relu" + str(i+1)))
            model.add(MaxPooling2D(pool_size=(1, 2)))

        model.add(Flatten())

        # add dense layers
        for i in range(hp.Int('dense_layers', 2, 4)):
            model.add(Dense(hp.Int('dense' + str(i) + '_filters',
                                   min_value=32,
                                   max_value=160,
                                   step=32),
                            activation='selu',
                            kernel_initializer="he_normal",
                            name="dense" + str(i)))
            model.add(AlphaDropout(hp.Float('dense' + str(i) + '_dr',
                                            min_value=0.1,
                                            max_value=0.5,
                                            step=0.1)))
        model.add(Dense(self.target_num, name="output"))
        model.add(Activation('softmax'))
        model.add(Reshape([self.target_num]))

        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=Adam(
                          hp.Choice(
                              'learning_rate',
                              values=[1e-4, 1e-3, 1e-2])))
        return model
