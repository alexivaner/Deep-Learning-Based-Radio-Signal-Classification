#!/usr/bin/env python
from gnuradio import gr, blocks, analog
# import mediatools
import numpy as np


class source_alphabet(gr.hier_block2):
    def __init__(self, dtype="discrete", limit=10000, randomize=False):
        if(dtype == "discrete"):
            gr.hier_block2.__init__(self, "source_alphabet",
                                    gr.io_signature(0, 0, 0),
                                    gr.io_signature(1, 1, gr.sizeof_char))

            self.src = blocks.file_source(
                gr.sizeof_char, "../source_material/gutenberg_shakespeare.txt")
            self.convert = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST)
            # self.convert = blocks.packed_to_unpacked_bb(8, gr.GR_LSB_FIRST);
            self.limit = blocks.head(gr.sizeof_char, limit)
            self.connect(self.src, self.convert)
            last = self.convert

            # whiten our sequence with a random block scrambler (optionally)
            if(randomize):
                rand_len = 256
                rand_bits = np.random.randint(2, size=rand_len)
                self.randsrc = blocks.vector_source_b(rand_bits, True)
                self.xor = blocks.xor_bb()
                self.connect(self.randsrc, (self.xor, 1))
                self.connect(last, self.xor)
                last = self.xor

        elif(dtype == "continuous"):   # "type_continuous"
            gr.hier_block2.__init__(self, "source_alphabet",
                                    gr.io_signature(0, 0, 0),
                                    gr.io_signature(1, 1, gr.sizeof_float))
            self.src = blocks.wavfile_source(
                '../source_material/serial-s01-e01.wav', True)
            self.float_short = blocks.float_to_short(1, 1)
            self.convert2 = blocks.interleaved_short_to_complex()
            self.convert3 = blocks.multiply_const_cc(1.0/65535)
            self.convert = blocks.complex_to_float()
            self.limit = blocks.head(gr.sizeof_float, limit)
            self.connect(self.src, self.float_short,
                         self.convert2, self.convert3, self.convert)
            last = self.convert
        elif(dtype == "noise"):   # "type_noise"
            gr.hier_block2.__init__(self, "source_alphabet",
                                    gr.io_signature(0, 0, 0),
                                    gr.io_signature(1, 1,
                                                    gr.sizeof_gr_complex))
            self.src = analog.fastnoise_source_c(analog.GR_GAUSSIAN,
                                                 1e-4, 0, 8192)
            self.limit = blocks.head(gr.sizeof_gr_complex, limit)
            last = self.src

        # connect head or not, and connect to output
        if(limit is None):
            self.connect(last, self)
        else:
            self.connect(last, self.limit, self)


if __name__ == "__main__":
    print("QA...")

    # Test discrete source
    tb = gr.top_block()
    src = source_alphabet("discrete", 1000)
    snk = blocks.vector_sink_b()
    tb.run()

    # Test continuous source
    tb = gr.top_block()
    src = source_alphabet("continuous", 1000)
    snk = blocks.vector_sink_f()
    tb.run()
