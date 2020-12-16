#!/usr/bin/env python
from transmitters import transmitters
from source_alphabet import source_alphabet
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft
import h5py
import random

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True

nvecs_per_key = 4096
vec_length = 128
snr_vals = range(-20, 30, 2)
global_modidx = 0

# count modulations  
mods_num = len(transmitters["continuous"]) + len(transmitters["discrete"]) + \
           len(transmitters["noise"])

dataset_x = np.zeros((mods_num * len(snr_vals) * nvecs_per_key, vec_length, 2), dtype=np.float32)
dataset_y = np.zeros((mods_num * len(snr_vals) * nvecs_per_key, mods_num), dtype=np.int64)
dataset_z = np.zeros((mods_num * len(snr_vals) * nvecs_per_key, 1), dtype=np.int64)

f = open("classes.txt", "w")
f.write("classes = [")

gindx = 0

for alphabet_type in transmitters.keys():
    for i, mod_type in enumerate(transmitters[alphabet_type]):
        if alphabet_type == "continuous" or alphabet_type == "discrete":
            if alphabet_type == "continuous":
                i = len(transmitters["discrete"]) + i
            f.write("'" + mod_type.modname + "',\n")
        elif alphabet_type == "noise":
            i = len(transmitters["discrete"]) + len(transmitters["continuous"])
            + i
            f.write("'" + mod_type.modname + "'")
        print(i, mod_type)
        for snr_idx, snr in enumerate(snr_vals):
            # moar vectors!
            modvec_indx = 0
            insufficient_modsnr_vectors = True
            while insufficient_modsnr_vectors:
                tx_len = int(10e3)
                if mod_type.modname == "QAM16":
                    tx_len = int(20e3)
                if mod_type.modname == "QAM64":
                    tx_len = int(30e3)
                src = source_alphabet(alphabet_type, tx_len, True)
                mod = mod_type()
                fD = 1
                delays = [0.0, 0.9, 1.7]
                mags = [1, 0.8, 0.3]
                ntaps = 8
                noise_amp = 10**(-snr/10.0)

                snk = blocks.vector_sink_c()

                tb = gr.top_block()
                if apply_channel:
                    chan = channels.dynamic_channel_model(
                        200e3, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays,
                        mags, ntaps, noise_amp, 0x1337)
                    if alphabet_type == "noise":
                        tb.connect(src, chan, snk)
                    else:
                        tb.connect(src, mod, chan, snk)
                else:
                    if alphabet_type == "noise":
                        tb.connect(src, snk)
                    else:
                        tb.connect(src, mod, snk)
                tb.run()

                raw_output_vector = np.array(snk.data(), dtype=np.complex64)

                # start the sampler some random time after channel model
                # transients (arbitrary values here)
                sampler_indx = random.randint(50, 500)
                while sampler_indx + vec_length < len(raw_output_vector) and \
                        modvec_indx < nvecs_per_key:
                    sampled_vector = raw_output_vector[
                        sampler_indx:sampler_indx+vec_length]
                    # Normalize the energy in this vector to be 1
                    energy = np.sum((np.abs(sampled_vector)))
                    sampled_vector = sampled_vector / energy
                    dataset_x[gindx, :, 0] = np.real(sampled_vector)
                    dataset_x[gindx, :, 1] = np.imag(sampled_vector)
                    dataset_y[gindx, :] = np.zeros((1, 11))
                    dataset_y[gindx, i] = 1
                    dataset_z[gindx, 0] = snr
                    # bound the upper end very high so it's likely we get
                    # multiple passes through independent channels

                    if round(len(raw_output_vector) * 0.05) < vec_length:
                        sampler_indx += vec_length
                    else:
                        sampler_indx += random.randint(vec_length,
                            round(len(raw_output_vector) * 0.05))
                    modvec_indx += 1
                    gindx += 1

                    # if ((modvec_indx % 500) == 0):
                    #     print("example: " + str(modvec_indx))

                if modvec_indx == nvecs_per_key:
                    insufficient_modsnr_vectors = False

f.write("]\n")
dataset = h5py.File('SIGNN_2019_01.hdf5', 'w')
dataset.create_dataset('X', data=dataset_x)
dataset.create_dataset('Y', data=dataset_y)
dataset.create_dataset('Z', data=dataset_z)
dataset.close()
