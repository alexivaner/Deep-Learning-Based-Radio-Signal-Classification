#!/usr/bin/env python
from gnuradio import gr, blocks, digital, analog, filter
sps = 8
ebw = 0.35


class transmitter_mapper(gr.hier_block2):
    def __init__(self, modtype, txname, samples_per_symbol=2,
                 excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        # self.mod = mapper.mapper(modtype, symvals)
        self.mod = modtype
        # pulse shaping filter
        nfilts = 32
        # make nfilts filters of ntaps each
        ntaps = nfilts * 11 * int(samples_per_symbol)
        rrc_taps = filter.firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            excess_bw,  # excess bandwidth (roll-off factor)
            ntaps)
        self.rrc_filter = filter.pfb_arb_resampler_ccf(
            samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)


class transmitter_bpsk(transmitter_mapper):
    modname = "BPSK"

    def __init__(self):
        modtype = digital.psk.psk_mod(
            constellation_points=2,
            mod_code="gray",
            differential=True,
            samples_per_symbol=sps,
            excess_bw=ebw,
            verbose=False,
            log=False,
        )
        transmitter_mapper.__init__(self, modtype, "transmitter_bpsk", sps,
                                    ebw)


class transmitter_qpsk(transmitter_mapper):
    modname = "QPSK"

    def __init__(self):
        modtype = digital.psk.psk_mod(
            constellation_points=4,
            mod_code="gray",
            differential=True,
            samples_per_symbol=sps,
            excess_bw=ebw,
            verbose=False,
            log=False,
        )
        transmitter_mapper.__init__(self, modtype, "transmitter_qpsk", sps,
                                    ebw)


class transmitter_8psk(transmitter_mapper):
    modname = "8PSK"

    def __init__(self):
        modtype = digital.psk.psk_mod(
            constellation_points=8,
            mod_code="gray",
            differential=True,
            samples_per_symbol=sps,
            excess_bw=ebw,
            verbose=False,
            log=False,
        )
        transmitter_mapper.__init__(self, modtype, "transmitter_8psk", sps,
                                    ebw)


class transmitter_qam16(transmitter_mapper):
    modname = "QAM16"

    def __init__(self):
        modtype = digital.qam.qam_mod(
            constellation_points=16,
            mod_code="gray",
            differential=True,
            samples_per_symbol=2,
            excess_bw=0.35,
            verbose=False,
            log=False,
        )
        transmitter_mapper.__init__(self, modtype, "transmitter_qam16", sps,
                                    ebw)


class transmitter_qam64(transmitter_mapper):
    modname = "QAM64"

    def __init__(self):
        modtype = digital.qam.qam_mod(
            constellation_points=64,
            mod_code="gray",
            differential=True,
            samples_per_symbol=2,
            excess_bw=0.35,
            verbose=False,
            log=False,
        )
        transmitter_mapper.__init__(self, modtype, "transmitter_qam64", sps,
                                    ebw)


class transmitter_gfsk(gr.hier_block2):
    modname = "GFSK"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_gfsk",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.repack = blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST)
        self.mod = digital.gfsk_mod(sps, sensitivity=0.1, bt=ebw)
        self.connect(self, self.repack, self.mod, self)


class transmitter_cpfsk(gr.hier_block2):
    modname = "CPFSK"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_cpfsk",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = analog.cpfsk_bc(0.5, 1.0, sps)
        self.connect(self, self.mod, self)


class transmitter_fm(gr.hier_block2):
    modname = "WBFM"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_fm",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = analog.wfm_tx(audio_rate=44100.0, quad_rate=220.5e3)
        self.connect(self, self.mod, self)
        self.rate = 200e3/44.1e3


class transmitter_am(gr.hier_block2):
    modname = "AM-DSB"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_am",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = 44.1e3/200e3
        # self.rate = 200e3/44.1e3
        self.interp = filter.fractional_resampler_ff(0.0, self.rate)
        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_cc(1.0)
        self.add = blocks.add_const_cc(1.0)
        self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_cc()
        self.connect(self, self.interp, self.cnv,
                     self.mul, self.add, self.mod, self)
        self.connect(self.src, (self.mod, 1))


class transmitter_amssb(gr.hier_block2):
    modname = "AM-SSB"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_amssb",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = 44.1e3/200e3
        # self.rate = 200e3/44.1e3
        self.interp = filter.fractional_resampler_ff(0.0, self.rate)
        self.mul = blocks.multiply_const_ff(1.0)
        self.add = blocks.add_const_ff(1.0)
        self.src = analog.sig_source_f(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_ff()
        self.filt = filter.hilbert_fc(401)
        self.connect(self, self.interp, self.mul,
                     self.add, self.mod, self.filt, self)
        self.connect(self.src, (self.mod, 1))


class transmitter_noise(gr.hier_block2):
    modname = "NOISE"

    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_noise",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        # self.mod = analog.fastnoise_source_c(analog.GR_GAUSSIAN, 1e-4, 0, 8192)
        # self.connect(self, self.mod, self)


transmitters = {
    "discrete": [transmitter_bpsk, transmitter_qpsk, transmitter_8psk,
                 transmitter_qam16, transmitter_qam64, transmitter_gfsk,
                 transmitter_cpfsk],
    "continuous": [transmitter_fm, transmitter_am, transmitter_amssb],
    "noise": [transmitter_noise]
}
