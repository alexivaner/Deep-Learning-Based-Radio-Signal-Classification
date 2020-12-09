"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import tensorflow.python.keras.models as models
from tensorflow.python.keras.layers import (Reshape, Dense, Dropout,
                                            Flatten)
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          ZeroPadding2D)
from tensorflow.python.keras.optimizers import Adam

from kerastuner import HyperModel


class deepsigcnn(HyperModel):

    def __init__(self, input_shape, target_num):
        self.input_shape = input_shape
        self.target_num = target_num

    def build(self, hp):
        model = models.Sequential()
        model.add(Reshape(self.input_shape + [1],
                          input_shape=self.input_shape))
        # 1st conv layer
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(hp.Int('conv1_filters',
                                min_value=64,
                                max_value=128,
                                step=32),
                         (1, 3),
                         activation='relu',
                         name='conv1',
                         padding='valid',
                         kernel_initializer='glorot_uniform'))
        model.add(Dropout(hp.Float('conv1_dr',
                                   min_value=0.1,
                                   max_value=0.5,
                                   step=0.1)))

        # 2nd conv layer
        model.add(ZeroPadding2D((0, 2)))
        model.add(Conv2D(hp.Int('conv2_filters',
                                min_value=32,
                                max_value=96,
                                step=32),
                         (1, 3),
                         activation='relu',
                         name='conv2',
                         padding='valid',
                         kernel_initializer='glorot_uniform'))
        model.add(Dropout(hp.Float('conv2_dr',
                                   min_value=0.1,
                                   max_value=0.5,
                                   step=0.1)))

        model.add(Flatten())

        # 1st dense layer
        model.add(Dense(hp.Int('dense1_filters',
                               min_value=128,
                               max_value=256,
                               step=128),
                        activation='relu',
                        name='dense1'))
        model.add(Dropout(hp.Float('dense1_dr',
                                   min_value=0.1,
                                   max_value=0.5,
                                   step=0.1)))
        # 2nd dense layer
        model.add(Dense(self.target_num,
                        activation='softmax',
                        name='dense2'))
        model.add(Reshape([self.target_num]))
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=Adam(
                          hp.Choice(
                              'learing_rate',
                              values=[1e-4, 1e-3, 1e-2])))
        return model
