"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import numpy as np
import os
import errno
import h5py


class dataset_generator():
    """
    A toolkit for parsing the signal dataset stored in HDF5 format.
    The structure of the HDF5 dataset should follow the structure as described
    above:

    HDF5 Dataset Structure
    -----------------------
    Group '/'
    Dataset 'X'
        Size:  2x1024xN
        Datatype:   FLOAT 32
    Dataset 'Y'
        Size:  25xN
        MaxSize:  25xN
        Datatype:   FLOAT 64
    Dataset 'Z'
        Size:  1xN
        Datatype:   INT 64

    Attributes
    ----------
    dataset_path : string
        the path to the directory of the dataset
    dataset_name : string
        the name including the extension of the model file
    modulation: string
        list of strings that defines a subset of samples with specific
        modulation samples from the selected dataset
    snr: int
        list of integers that defines a subset of samples with specific SNR
        from the selected dataset
    split_ratio : float
        a triplet of floats that define the train/validation/test portions of
        the dataset. Default: [0.8 0.1 0.1]
    """

    def __init__(self, dataset_path, dataset_name,
                 dataset_percent=1,
                 data_transform='cartesian',
                 modulation=None, snr=None,
                 split_ratio=[0.8, 0.1, 0.1]):
        self.__init_dataset_path(dataset_path)
        self.dataset = h5py.File(self.dataset_path+dataset_name,
                                 'r')
        self.__init_data()
        self.__init_modulations(modulation)
        self.__init_snr()
        self.snr_num = len(np.unique(self.snr))
        self.samples_per_snr_mod = self = \
            int(np.sum(self.modulations, axis=0)[0] / self.snr_num)
        self.dpercarg = dataset_percent
        self.modarg = modulation
        self.snrarg = snr
        self.mod_classes = self.list_available_modulations()
        self.total_samples_num = self.get_total_samples()
        # TODO: Add check for the split ratio list dimension
        self.split_ratio = split_ratio
        self.train_samples = int(split_ratio[0] * self.samples_per_snr_mod)
        self.valid_samples = int(split_ratio[1] * self.samples_per_snr_mod)
        self.test_samples = int(split_ratio[2] * self.samples_per_snr_mod)

    def __init_dataset_path(self, path):
        if (not os.path.exists(path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    path)
        self.dataset_path = os.path.join(path, '')

    def __init_data(self):
        """
        A method to initialize the samples dataset from the HDF5 file.
        """
        self.data = self.dataset['X']

    def __init_modulations(self, modulation):
        """
        A method to initialize the targets dataset from the HDF5 file.
        """
        self.modulations = self.dataset['Y']
        # If modulation subset is given as argument count classes number
        if modulation is not None:
            self.mods_num = len(modulation)
        # Else get default classes subset from local classes.txt file
        else:
            self.mods_num = len(self.list_available_modulations())

    def __init_snr(self):
        """
        A method to initialize the SNR dataset from the HDF5 file.
        """
        self.snr = self.dataset['Z']

    def __get_index(self, occur, l):
        return [l.index(idx.upper()) for idx in occur]

    def list_selected_modulations(self):
        if self.modarg is not None:
            return [m.upper() for m in self.modarg]
        else:
            self.list_available_modulations()

    def list_available_modulations(self):
        """
        The HDF5 dataset is followed by a txt file that defines the number and
        names of the classes contained in the dataset.

        TXT Structure
        -------------------
        classes = ['MOD1',
                   'MOD2',
                   ....
                   'MODn']
        """
        f = open(self.dataset_path+'/classes.txt', 'r')
        s = f.read().replace(" ", "")
        class_list = s.split("',\n'")
        class_list[0] = class_list[0].split("classes=['")[1]
        class_list[-1] = class_list[-1].split("']\n")[0]
        return class_list

    def get_total_samples(self):
        """
        Calculate the total number of samples from the initial dataset as
        results based on the SNR and modulation subsets requested.
        """
        if self.modarg is not None and self.snrarg is not None:
            return len(self.modarg)*len(self.snrarg)*self.samples_per_snr_mod
        elif self.modarg is None:
            return (len(self.snrarg) *
                    self.mods_num *
                    self.samples_per_snr_mod)
        elif self.snrarg is None:
            return len(self.modarg)*self.snr_num*self.samples_per_snr_mod
        else:
            return self.modulations[:, 0].size

    def transform_example(self, x, transform):
        if transform == 'cartesian':
            return x
        elif transform == 'polar':
            y = np.zeros(x.shape, dtype=np.float32)
            y[0, :] = np.sqrt(x[0, :]**2 + x[1, :]**2)
            y[1, :] = np.arctan2(x[1, :], x[0, :])
            return y
        else:
            raise ValueError('Invalid data transformation')

    def test_dataset_generator(self):
        """
        Generator for the test dataset.
        """
        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(
                self.train_samples+self.valid_samples,
                    self.train_samples+self.valid_samples +
                    int(self.test_samples * self.dpercarg)):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    cur_example = self.data[indx[j, i], :, :].transpose()
                    yield (sself.transform_example(cur_example,
                                                   self.data_transform),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])

    def validation_dataset_generator(self):
        """
        Generator for the validation dataset.
        """
        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(self.train_samples,
                           self.train_samples +
                           int(self.valid_samples * self.dpercarg)):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    cur_example = self.data[indx[j, i], :, :].transpose()
                    yield (self.transform_example(cur_example,
                                                  self.data_transform),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])

    def train_dataset_generator(self):
        """
        Generator for the train dataset.
        """
        if self.snrarg is None:
            self.snrarg = [i for i in range(-20, 32, 2)]

        if self.modarg is None:
            self.modarg = self.list_available_modulations()

        mods = self.__get_index(self.modarg, self.mod_classes)
        for column in (self.snr[:] == self.snrarg).T:
            q = np.where(column &
                         (self.modulations[:, mods] == 1).any(axis=1))[0]
            indx = (np.array(q).reshape(-1, 1))
            indx = indx.reshape(len(mods), self.samples_per_snr_mod)

            for i in range(0, int(self.train_samples*self.dpercarg)):
                # Pop every modulation sample
                for j in range(0, indx.shape[0]):
                    cur_example = self.data[indx[j, i], :, :].transpose()
                    yield (self.transform_example(cur_example,
                                                  self.data_transform),
                           np.argmax(self.modulations[indx[:, i], :],
                                     axis=1)[j])
