#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 16/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################

import numpy as np
import simpleaudio as sa
import yaml
from box import Box
import coding


class TXtag:

    def __init__(self, CR=4, mode=0):
        self.mode = mode
        self.preamble_mask = np.array([-1, -1, -1, -1, -1, 1, 1, 1])
        self.CR=CR

        with open("config.yml", "r") as ymlfile:
            self.cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)
        self.print_tx_info()

        self.preamble = self.create_preamble_packet()

    def create_symbols(self, text_message):
        # raw symbols
        print('Text:', text_message)
        symbols = self.text_to_numbers(text_message)
        raw_symbols = self.text_to_numbers(text_message)
        print('Initial Symbols:', symbols)
        header_symbols = np.array([symbols.size, self.CR])
        print('Header Symbols:', header_symbols)

        # %%%%%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%%%%%%%%
        # signal = self.lora_symbol(3, inverse=0)
        # self.send_packet_to_sound_card(signal)
        if self.mode == 1:
            # inverse Gray encoding for Payload
            for i in range(len(symbols)):
                symbols[i] = self.inverse_gray_encoder(symbols[i])
            print('Gray Payload Symbols:', symbols)

            for i in range(len(header_symbols)):
                header_symbols[i] = self.inverse_gray_encoder(header_symbols[i])
            print('Gray Header Symbols:', header_symbols)
        return header_symbols, symbols, raw_symbols

    def print_tx_info(self):
        print("%%%%%%%%%%%% TX Parameters  %%%%%%%%%%%%%%%%")
        print("Sample Rate:", self.cfg.TX_SAMPLING_RATE, "Sps")
        print("Spreading factor:", self.cfg.SF)
        print("Coding rate:", self.cfg.CR)
        print("BandWith:", self.cfg.BW, "Hz")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def inverse_gray_encoder(self, n):
        inv = 0
        # Taking xor until
        # n becomes zero
        while n:
            inv = inv ^ n
            n = n >> 1
        return inv

    def text_to_numbers(self, text):
        """
        Convert a string message to an integer number according the utf-8 encoding
        return: int numpy array
        """
        # print("The original message is : " + str(text))
        arr = bytes(text, 'utf-8')
        numbers = np.zeros(len(arr), dtype=int)
        i = 0
        for byte in arr:
            numbers[i] = byte
            i = i + 1
        return numbers

    def text_to_binary(self):
        text = self.message_text
        print("The original message is : " + str(text))
        sf = self.cfg.SF
        # using join() + ord() + format()
        # Converting String to binary
        res = ''.join(format(ord(i), '08b') for i in text)
        # printing result

        print("The message after binary conversion : " + str(res))
        # print("The message after dec conversion : " + arr2)
        return res

    def lora_symbol(self, shift, inverse=0):
        # Initialization
        phase = 0
        Frequency_Offset = self.cfg.CHIRP_F_START
        sf = self.cfg.SF
        fs = self.cfg.TX_SAMPLING_RATE
        bw = self.cfg.BW
        num_samples_in = fs * (2 ** sf) / bw
        num_samples = round(num_samples_in)

        signal = np.zeros((num_samples, 2))

        for k in range(num_samples):
            # set output to cosine signal
            signal[k, 0] = np.cos(phase)
            signal[k, 1] = np.sin(phase)

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

    def create_preamble_packet(self):
        ##Preamble###################################################
        mask = self.preamble_mask
        preamble_up_chirp = self.lora_symbol(0, 0)
        preamble_down_chirp = self.lora_symbol(0, 1)

        preamble = np.zeros((1, 2))
        for k in range(mask.size):
            if mask[k] == 1:
                preamble = np.concatenate((preamble, preamble_up_chirp), axis=0)
            elif mask[k] == -1:
                preamble = np.concatenate((preamble, preamble_down_chirp), axis=0)
            else:
                preamble_symbol_chirp = self.lora_symbol(mask[k], 1)
                preamble = np.concatenate((preamble, preamble_symbol_chirp), axis=0)
        preamble = preamble[1:, :]
        print("Preamble created, Matrix Dims:", preamble.shape)
        return preamble

    def create_packet(self, header_symbols, payload_symbols):

        ##Header#########################################################
        header = np.zeros((1, 2))
        for k in range(header_symbols.size):
            header_signal = self.lora_symbol(header_symbols[k], 0)
            header = np.concatenate((header, header_signal), axis=0)
        header = header[1:, :]

        ##Payload#########################################################
        payload = np.zeros((1, 2))
        for k in range(payload_symbols.size):
            payload_signal = self.lora_symbol(payload_symbols[k], 0)
            payload = np.concatenate((payload, payload_signal), axis=0)
        payload = payload[1:, :]
        print("Payload created Matrix Dims:", payload.shape)

        packet = np.concatenate((self.preamble, header), axis=0)
        packet = np.concatenate((packet, payload), axis=0)
        print("Packet created Matrix Dims:", packet.shape)

        return packet

    def send_packet_to_sound_card(self, packet):
        # This function uses the sound card in order to produce the chirp signals
        # packet is a array of floats
        sound_fs = self.cfg.TX_SAMPLING_RATE
        sound_res = self.cfg.SOUND_RES
        ch = 2
        if sound_res == 16:
            # Ensure that highest value is in 16-bit range
            audio = packet * (2 ** 15 - 1) / np.max(np.abs(packet))
            # Convert to 16-bit data
            audio = audio.astype(np.int16)
            play_obj = sa.play_buffer(audio, ch, 2, sound_fs)
            play_obj.wait_done()
        elif sound_res == 24:
            # normalize to 24-bit range
            packet *= (2 ** 23 - 1) / np.max(np.abs(packet))
            # convert to 32-bit data
            audio = packet.astype(np.int32)
            # convert from 32-bit to 24-bit by building a new byte buffer, skipping every fourth bit
            # note: this also works for 2-channel audio
            i = 0
            byte_array = []
            for b in audio.tobytes():
                if i % 4 != 3:
                    byte_array.append(b)
                i += 1
            audio = bytearray(byte_array)
            play_obj = sa.play_buffer(audio, ch, 3, sound_fs)
            play_obj.wait_done()
        # Start playback
        # Wait for playback to finish before exiting

        print("Packet sent to Sound Card")
