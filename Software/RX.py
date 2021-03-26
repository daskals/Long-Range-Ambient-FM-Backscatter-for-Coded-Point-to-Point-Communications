#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 16/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################

import numpy as np
import scipy.signal as sps
import yaml
from box import Box
import samplerate
import coding


class RX_receiver:

    def __init__(self, rf_samples, mode=0, num_symbols=0):

        with open("config.yml", "r") as ymlfile:
            self.cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)
        self.print_rx_info()

        # Convert samples to a numpy array
        # self.x1 = np.array(rf_samples).astype("complex64")
        # self.downconvert_signal()

        preamble_mask = np.array([-1, -1, -1, -1, -1, 1, 1, 1])
        self.symbols_per_preamble_header = preamble_mask.size + 2

        self.symbols_per_frame = len(preamble_mask) + num_symbols
        print(self.symbols_per_frame)
        self.T_symbol = (2 ** self.cfg.SF) / self.cfg.BW
        self.total_packet_duration = self.symbols_per_frame * self.T_symbol

        ##################################################################
        self.rf_signal = rf_samples[:, 0] + 1j * rf_samples[:, 1]

        self.down_signal = self.downconvert_signal(self.rf_signal)
        self.dec_signal = self.resample(self.down_signal)
        Fs = self.cfg.BW
        downChirp_serie = self.chirp_maker(Fs, len(self.dec_signal))
        # dechirp of the packet
        self.dechirp = self.dec_signal * downChirp_serie

        preamble_header_symbols = self.fft_decode_preample_header(Fs)
        if mode == 1:
            self.payload_length = self.gray_encoder(int(preamble_header_symbols[-2]))
            self.CR = self.gray_encoder(int(preamble_header_symbols[-1]))
        else:
            self.payload_length = int(preamble_header_symbols[-2])
            self.CR = int(preamble_header_symbols[-1])

        self.rx_out_symbols = self.fft_decode_v2(Fs, preamble_header_symbols)
        # self.rx_out_symbols = self.fft_decode(Fs, len(preamble_mask))

        if mode == 1:
            self.gray_rx_symbols = np.zeros(len(self.rx_out_symbols), dtype=int)
            for i in range(len(self.rx_out_symbols)):
                self.gray_rx_symbols[i] = self.gray_encoder(int(self.rx_out_symbols[i]))
            print("Gray DeCoding...symbols", self.gray_rx_symbols)

        # self.return_signal()

    ##def preamble_synchronization(self):
    #  corelation for preamble finding
    # np.correlate(arr1, arr2, "full")
    # signalStartIndex = abs(lag(cLag)) + 9 * params.T_symbol * Fs
    # signalStartIndex_time = signalStartIndex / Fs
    # signalEndIndex = round(signalStartIndex + symbols_per_frame*symbol_time*Fs)
    # signalEndIndex_time=signalEndIndex/Fs
    # sync_signal = resampled_signal[signalStartIndex:signalEndIndex]

    def return_rx_symbols(self, mode):
        if mode == 1:
            return self.gray_rx_symbols
        return self.rx_out_symbols

    def fft_decode_preample_header(self, Fs):
        over = int(self.T_symbol * Fs)
        symbols = np.zeros(self.symbols_per_preamble_header)  # create
        for m in range(self.symbols_per_preamble_header):
            signal = self.dechirp[(m * over): (m + 1) * over]
            FFT_out = np.abs(np.fft.fft(signal))
            r = np.max(FFT_out)
            c = np.argmax(FFT_out)
            print("Symbol#:\%d, Symbol:%d, Power:%d", m, c, r)
            symbols[m] = c
        symbols = symbols - round(np.mean(symbols[5:7])) % (2 ** self.cfg.SF)
        print("###################END Preamble header")
        return symbols

    def fft_decode_v2(self, Fs, symbols_pr_header):
        over = int(self.T_symbol * Fs)
        symbols = np.zeros(self.payload_length)  # create
        for m in range(self.payload_length):
            signal = self.dechirp[((m + self.symbols_per_preamble_header) * over): ((
                                                                                                m + self.symbols_per_preamble_header) + 1) * over]
            FFT_out = np.abs(np.fft.fft(signal))
            r = np.max(FFT_out)
            c = np.argmax(FFT_out)
            print("Symbol#:\%d, Symbol:%d, Power:%d", m, c, r)
            symbols[m] = c
        symbols = symbols - round(np.mean(symbols_pr_header[5:7])) % (2 ** self.cfg.SF)
        return symbols

    def fft_decode(self, Fs, len_preamble):
        over = int(self.T_symbol * Fs)
        symbols = np.zeros(self.symbols_per_frame)  # create
        for m in range(self.symbols_per_frame):
            signal = self.dechirp[(m * over): (m + 1) * over]
            FFT_out = np.abs(np.fft.fft(signal))
            r = np.max(FFT_out)
            c = np.argmax(FFT_out)
            print("Symbol#:\%d, Symbol:%d, Power:%d", m, c, r)
            symbols[m] = c
        symbols = symbols - round(np.mean(symbols[5:7])) % (2 ** self.cfg.SF)
        RX_symbols = symbols[len_preamble:]
        print(RX_symbols)
        return RX_symbols

    def return_signal(self):
        return self.dec_signal

    def downconvert_signal(self, rf_signal):
        F_offset = self.cfg.CHIRP_F_START + self.cfg.BW / 2
        # To mix the data down, generate a digital complex exponential
        # (with the same length as x1) with phase -F_offset/Fs
        fc1 = np.exp(-1.0j * 2.0 * np.pi * F_offset / self.cfg.TX_SAMPLING_RATE * np.arange(len(self.rf_signal)))
        # Now, just multiply x1 and the digital complex expontential
        x_down = rf_signal * fc1
        return x_down

    def resample(self, downconv_signal):

        newrate = self.cfg.BW
        # channelize the signal
        ########################################################################
        samples = round(downconv_signal.size * newrate / self.cfg.TX_SAMPLING_RATE)
        resampled_signal1 = sps.resample(downconv_signal, samples)
        ########################################################################
        dec_audio = int(self.cfg.TX_SAMPLING_RATE / newrate)
        resampled_signal2 = sps.decimate(downconv_signal, dec_audio)
        ########################################################################
        # ratio = newrate / self.cfg.TX_SAMPLING_RATE
        # converter = 'sinc_best'  # or 'sinc_fastest', ...
        # resampled_signal3 = samplerate.resample(downconv_signal, ratio, converter)
        return resampled_signal1

    def chirp_maker(self, Fs, length):

        f0 = -self.cfg.BW / 2
        downChirp = self.my_chirp(Fs, 1, f0)
        # downChirp=self.lora_symbol(self.cfg.SF, self.cfg.BW, Fs, 0, 1, f0, 3)
        downChirp_series = np.tile(downChirp, int(np.ceil(length / len(downChirp))))
        downChirp_series = downChirp_series[:length]
        return downChirp_series

    def my_chirp(self, fs, inverse, left):
        sf = self.cfg.SF
        bw = self.cfg.BW

        symbol_time = 2 ** sf / bw
        k = bw / symbol_time

        t = np.arange(0, symbol_time, 1 / fs)
        if inverse == 0:
            if_chirp = 2 * np.pi * (left * t + k / 2 * t ** 2)
        elif inverse == 1:
            if_chirp = -2 * np.pi * (left * t + k / 2 * t ** 2)
        else:
            print("Error argument")
        chirp_out = np.cos(if_chirp) + 1j * np.sin(if_chirp)
        chirp_out = (1 + 1j) * chirp_out
        return chirp_out

    def print_rx_info(self):
        print("%%%%%%%%%%%% RX Parameters  %%%%%%%%%%%%%%%%")
        print("Sample Rate:", self.cfg.TX_SAMPLING_RATE, "Sps")
        print("Spreading factor:", self.cfg.SF)
        print("Coding rate:", self.cfg.CR)
        print("BandWith:", self.cfg.BW, "Hz")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def gray_encoder(self, n):

        """
        Convert given decimal number into decimal equivalent of its gray code form
        :param n: decimal number
        return: int
        """
        # Right Shift the number by 1 taking xor with original number
        return n ^ (n >> 1)

    def text_to_numbers(self):
        """
        Convert a string message to an integer number according the utf-8 encoding
        return: int numpy array
        """
        text = self.message_text
        # print("The original message is : " + str(text))
        arr = bytes(text, 'utf-8')
        numbers = np.zeros(len(arr), dtype=int)
        i = 0
        for byte in arr:
            numbers[i] = byte
            i = i + 1
        return numbers

    def lora_symbol(self, sf, bw, fs, symbol, inverse, left, mode):
        # Initialization
        phase = 0
        Frequency_Offset = left
        shift = symbol
        num_samples_in = fs * (2 ** sf) / bw
        num_samples = round(num_samples_in)

        if mode == 1 or mode == 2:
            signal = np.zeros(num_samples)
        elif mode == 3:
            signal = np.zeros(num_samples) + 0j

        for k in range(num_samples):
            # set output to cosine signal
            if mode == 1:
                signal[k] = np.cos(phase)
            elif mode == 2:
                signal[k] = np.sign(np.cos(phase))
            elif mode == 3:
                signal[k] = np.cos(phase) + 1j * np.sin(phase)
            # ------------------------------------------
            # Frequency from cyclic shift
            f = bw * shift / (2 ** sf)
            if inverse == 1:
                f = bw - f
            # ------------------------------------------
            # apply Frequency offset away from DC
            f = f + Frequency_Offset
            # ------------------------------------------
            # Increase the phase according to frequency
            phase = phase + 2 * np.pi * f / fs
            if phase > np.pi:
                phase = phase - 2 * np.pi
            # ------------------------------------------
            # update cyclic shift
            shift = shift + bw / fs
            if shift >= (2 ** sf):
                shift = shift - 2 ** sf
        return signal

    def rx_encoder(self, RX_symbols, raw_symbols_tx):
        in_bit_matrix = coding.vec_bin_array(raw_symbols_tx, self.cfg.SF)
        in_bit_vector = in_bit_matrix.reshape(-1)
        N_bits_raw = len(in_bit_vector)
        # Round to a whole number of interleaving blocks
        N_bits = int(np.ceil(N_bits_raw / (4 * self.cfg.SF)) * (4 * self.cfg.SF))
        N_codewords = int(N_bits / 4)
        N_codedbits = N_codewords * (4 + self.cfg.CR)
        N_syms = int(N_codedbits / self.cfg.SF)

        gray_rx_symbols = np.zeros(len(RX_symbols), dtype=int)
        for i in range(len(RX_symbols)):
            gray_rx_symbols[i] = self.gray_encoder(int(RX_symbols[i]))
        print("Gray DeCoding...symbols", gray_rx_symbols)
        bit_matrix = coding.vec_bin_array(gray_rx_symbols, self.cfg.SF)
        # print("Matrix degray\n", bit_matrix)
        # ------------------------------------------------------
        Cest = coding.de_inter_liver(bit_matrix, N_codewords, N_codedbits, self.cfg.CR, self.cfg.SF)
        # print("DeInterliving Coding Cest matrix\n", Cest)

        bits_est = coding.Hamming_decode(Cest, N_bits, N_codewords, self.cfg.CR)
        ints = 0

        bit_matrix_final = np.zeros((raw_symbols_tx.size, self.cfg.SF), dtype=np.int32)
        for sym in range(0, bits_est.size, self.cfg.SF):
            bit_matrix_final[ints, :] = bits_est[sym: sym + self.cfg.SF]
            a = bit_matrix_final[ints, :]
            ints = ints + 1

        # print("bit final\n", bit_matrix_final)
        symbols_matrix_final = [coding.bool2int(x[::-1]) for x in bit_matrix_final]
        print("binary\n", symbols_matrix_final)
        if np.array_equal(raw_symbols_tx, symbols_matrix_final):
            print("!!!!!!!!!!!!!!Correct Packet!!!!!!!!!!!!!!!!!")
        else:
            print("***************Wrong Packet******************")
