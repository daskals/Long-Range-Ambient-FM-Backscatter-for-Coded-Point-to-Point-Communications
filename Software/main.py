#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 19/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################
from matplotlib import pyplot as plt
from TX import TXtag
from RX import RX_receiver
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from messenger import Ui_MainWindow
import sys
import time
from pynput.keyboard import Key,Controller
keyboard = Controller()


if __name__ == '__main__':

    tag = TXtag(mode=1)

    header_symbols, symbols, tx_symbols = tag.create_symbols('Spiros')
    packet = tag.create_packet(header_symbols, symbols)

    rx = RX_receiver(packet, num_symbols=7, mode=1)
    rx_symbols = rx.return_rx_symbols(mode=1)

    print(tx_symbols)

    if np.array_equal(rx_symbols, tx_symbols):
        print("!!!!!!!!!!!!!!Correct Packet!!!!!!!!!!!!!!!!!")
    else:
        print("***************Wrong Packet******************")

    while True:
        for i in range(10):
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
            time.sleep(0.1)
        for i in range(10):
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
            time.sleep(0.1)
        time.sleep(2)
        # dec_signal = rx.return_signal()

    # fig = plt.figure(1)
    # NFFT = 1024
    # #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    # plt.specgram(packet[:,1], NFFT=NFFT, Fs=192000, noverlap=900)
    # plt.title('Spectrogram')
    # plt.ylabel('Frequency band')
    # plt.xlabel('Time window')
    # plt.grid(True)
    # plt.draw()
    # plt.show(block=True)
    # plt.savefig("RX_Packet.pdf")

    # fig = plt.figure(2)
    # NFFT = 1024
    # Fs_new=5000
    # # #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    # plt.specgram(dec_signal, NFFT=NFFT, Fs=Fs_new, noverlap=900)
    # plt.title('Spectrogram')
    # plt.ylabel('Frequency band')
    # plt.xlabel('Time window')
    # plt.grid(True)
    # plt.draw()
    # plt.show(block=True)
