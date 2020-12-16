"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import itertools
import os
import errno
import io


class plotter():

    def __init__(self, dest_path):
        self.__init_destination_path(dest_path)

    def __init_destination_path(self, dest_path):
        if (not os.path.exists(dest_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    dest_path)
        self.dest_path = os.path.join(dest_path, '')

    def plot_confusion_matrix(self, cm, class_names, filename):

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                       decimals=2)
        figure = plt.figure(figsize=(15, 15))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar(im)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.dest_path+filename,
                    bbox_inches='tight')
        return figure

    def plot_training_validation_loss(self, history, filename):
        figure = plt.figure()
        plt.title('Training Performance')
        plt.plot(history.epoch, history.history['loss'],
                 label='Train Loss + Error')
        plt.plot(history.epoch, history.history['val_loss'],
                 label='Validation Error')
        plt.legend()
        # plt.savefig(self.dest_path+filename,
        #             bbox_inches='tight')
        return figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
