"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

import argparse
import os
import errno
import struct
import h5py
from shutil import copyfile
import numpy as np


class generate_satnogs_noise_dataset():

    def __init__(self, source_dir, destination, samples_per_row, rows_num,
                 deepsig_path):
        self.__init_source_dir(source_dir)
        self.__init_dest_dir(destination)
        self.__init_deepsig_path(deepsig_path)
        self.channels_num = 2
        self.samples_per_row = samples_per_row
        self.rows_num = rows_num
        self.scale_factor = 32767.0

    def __init_source_dir(self, source_dir):
        if (not os.path.exists(source_dir)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    source_dir)
        self.source_dir = os.path.join(source_dir, '')

    def __init_dest_dir(self, dest_dir):
        if (not os.path.exists(dest_dir)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    dest_dir)
        self.destination = os.path.join(dest_dir, '')

    def __init_deepsig_path(self, deepsig_path):
        if deepsig_path != "":
            if (not os.path.isfile(deepsig_path)):
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        deepsig_path)
            self.deepsig_path = deepsig_path

    def __deinterleave(self, vec):
        return [vec[idx::self.channels_num]
                for idx in range(self.channels_num)]

    def __count_files(self):
        cnt = 0
        for filename in os.listdir(self.source_dir):
            if filename.startswith("iq_s16_"):
                cnt += 1
            else:
                continue
        return cnt

    def __iterate_source_dir(self):
        for filename in os.listdir(self.source_dir):
            if filename.startswith("iq_s16_"):
                yield os.path.join(self.source_dir, filename)
            else:
                continue

    def __expand_deepsig_dataset(self):
        copyfile(self.deepsig_path, self.destination + "deepsig.hdf5")
        fo = h5py.File(self.destination + "deepsig.hdf5", 'r+')
        y = fo["Y"]
        del fo["Y"]
        y = np.hstack([y, np.zeros((y.shape[0], 1))])
        fo.create_dataset("Y", data=y)
        fo.close()

    def __concat_datasets(self):
        '''
        Concatenate multiple files into a single virtual dataset
        '''
        # self.__expand_deepsig_dataset()
        fo = h5py.File(self.destination + "deepsig.hdf5", 'r+')
        fn = h5py.File(self.destination + "satnogs_noise.hdf5", 'r')

        x_layout = h5py.VirtualLayout(
            shape=(fo["X"].shape[0]+fn["X"].shape[0],
                   fn["X"].shape[1], fn["X"].shape[2]))

        vsource = h5py.VirtualSource(self.destination + "deepsig.hdf5",
                                     "X", shape=fo["X"].shape)
        x_layout[0:fo["X"].shape[0]] = vsource
        vsource = h5py.VirtualSource(self.destination + "satnogs_noise.hdf5",
                                     "X", shape=fn["X"].shape)
        x_layout[
            fo["X"].shape[0]:fo["X"].shape[0]+fn["X"].shape[0]] = vsource

        y_layout = h5py.VirtualLayout(
            shape=(fo["Y"].shape[0]+fn["Y"].shape[0], fn["Y"].shape[1]))

        vsource = h5py.VirtualSource(self.destination + "deepsig.hdf5",
                                     "Y", shape=fo["Y"].shape)
        y_layout[0:fo["Y"].shape[0]] = vsource
        vsource = h5py.VirtualSource(self.destination + "satnogs_noise.hdf5",
                                     "Y", shape=fn["Y"].shape)
        y_layout[
            fo["Y"].shape[0]:fo["Y"].shape[0]+fn["Y"].shape[0]] = vsource

        z_layout = h5py.VirtualLayout(
            shape=(fo["Z"].shape[0]+fn["Z"].shape[0], 1))

        vsource = h5py.VirtualSource(self.destination + "deepsig.hdf5",
                                     "Z", shape=fo["Z"].shape)
        z_layout[0:fo["Z"].shape[0]] = vsource
        vsource = h5py.VirtualSource(self.destination + "satnogs_noise.hdf5",
                                     "Z", shape=fn["Z"].shape)
        z_layout[
            fo["Z"].shape[0]:fo["Z"].shape[0]+fn["Z"].shape[0]] = vsource

        with h5py.File(self.destination+"VDS.h5", 'w', libver='latest') as f:
            f.create_virtual_dataset("X", x_layout, fillvalue=0)
            f.create_virtual_dataset("Y", y_layout, fillvalue=0)
            f.create_virtual_dataset("Z", z_layout, fillvalue=0)

    def generate_dataset(self):
        files_cnt = self.__count_files()
        rows_per_file = self.rows_num // files_cnt
        spare_rows = self.rows_num % files_cnt
        # Global row counter
        gcnt = 0

        x = np.zeros((self.rows_num, self.samples_per_row, 2), dtype="float32")
        y = np.zeros((self.rows_num, 25), dtype="uint8")
        y[:, -1] = 1
        z = np.full((self.rows_num, 1), -20, dtype="float32")
        for fn in self.__iterate_source_dir():
            with open(fn, 'rb') as f:
                # Skip some samples to avoid zeros
                f.seek(4*64)
                if spare_rows > 0:
                    for r in range(0, rows_per_file + 1):
                        for s in range(0, self.samples_per_row):
                            # SatNOGS observations are stored as
                            # interleaved uint16
                            raw = f.read(4)
                            if len(raw) != 4:
                                # ignore the incomplete "record" if any
                                break
                            iq = struct.unpack("HH", raw)
                            x[gcnt, s, 0] = iq[0] / self.scale_factor
                            x[gcnt, s, 1] = iq[1] / self.scale_factor
                        gcnt += 1
                    spare_rows -= 1
                else:
                    for r in range(0, rows_per_file):
                        for s in range(0, self.samples_per_row):
                            # SatNOGS observations are stored as
                            # interleaved uint16
                            raw = f.read(4)
                            if len(raw) != 4:
                                # ignore the incomplete "record" if any
                                break
                            iq = struct.unpack("hh", raw)
                            x[gcnt, s, 0] = iq[0] / self.scale_factor
                            x[gcnt, s, 1] = iq[1] / self.scale_factor
                        gcnt += 1
        # TODO: Check normalization again
        x = x/np.abs(x).max(keepdims=True, axis=1)
        dataset = h5py.File(self.destination + '/satnogs_noise.hdf5', 'w')
        dataset.create_dataset('X', data=x)
        dataset.create_dataset('Y', data=y)
        dataset.create_dataset('Z', data=z)
        dataset.close()
        self.__concat_datasets()


def argument_parser():
    description = 'A tool to generate models using Keras/Tensorflow'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument("-s", "--source-directory", dest="source_dir",
                        action='store',
                        help="The source directory of SatNOGS directory.")
    parser.add_argument("-d", "--destination-path", dest="destination",
                        action='store',
                        help="The full path of the exported HDF5 dataset.")
    parser.add_argument("-p", "--deepsig-dataset-path", dest="deepsig_path",
                        action='store', default="",
                        help="The full path to the Deepsig Inc dataset.")
    parser.add_argument("--samples-per-row", dest="samples_per_row",
                        action='store', type=int, default=1024,
                        help="The number of IQ samples per \
                            row in the dataset.")
    parser.add_argument("--rows-num", dest="rows_num", type=int,
                        action='store', default=4096,
                        help="The number of rows in the dataset.")
    return parser


def main(generator=generate_satnogs_noise_dataset, args=None):
    if args is None:
        args = argument_parser().parse_args()

    g = generator(source_dir=args.source_dir, destination=args.destination,
                  samples_per_row=args.samples_per_row, rows_num=args.rows_num,
                  deepsig_path=args.deepsig_path)
    g.generate_dataset()


if __name__ == '__main__':
    main()
