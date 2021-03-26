#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 19/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################
import asyncio

from matplotlib import pyplot as plt
from TX import TXtag
from RX import RX_receiver
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow
from messenger import Ui_MainWindow
import yaml
from box import Box
from datetime import datetime
from pynput.keyboard import Key, Controller
from rtlsdr import RtlSdr

import sys


async def streaming():
    sdr = RtlSdr()
    sample_rate = 1e6
    sdr.freq_correction = 60  # PPM
    sdr.gain = 'auto'
    t_sampling = 1  # Sampling
    N_samples = round(sample_rate * t_sampling)
    samples = sdr.read_samples(N_samples)

    async for samples in sdr.stream():
        print(samples)
    # ...

    # to stop streaming:
    await sdr.stop()

    # done
    sdr.close()





class Tag_transmitter(QObject):
    # This defines a signal called 'finished' that takes no arguments.
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(Tag_transmitter, self).__init__()
        self.working = True
        self.text = 'LORAB'
        self.tag = TXtag(mode=0)

    @pyqtSlot()
    def work(self):
        """Long-running task."""
        i = 0
        after_text = 0
        while self.working:
            # i = i + 1
            if self.text != after_text:
                # print('******************************TAG Running Loop**************************')
                header_symbols, symbols, tx_symbols = self.tag.create_symbols(self.text)
                packet = self.tag.create_packet(header_symbols, symbols)
                after_text = self.text

            # self.tag.send_packet_to_sound_card(packet)

        self.finished.emit()


class SDR_receiver(QObject):
    # This defines a signal called 'finished' that takes no arguments.
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(SDR_receiver, self).__init__()
        self.sdr = RtlSdr()
        self.working = True
        sample_rate = 1e6
        self.sdr.freq_correction = 60  # PPM
        self.sdr.gain = 'auto'
        t_sampling = 1  # Sampling
        self.N_samples = round(sample_rate * t_sampling)
        #self.loop = asyncio.get_event_loop()

    @pyqtSlot()
    def work(self):
        """Long-running task."""
        i = 0

        #self.run_until_complete(streaming())

        while self.working:
            #samples = self.sdr.read_samples(self.N_samples)
            #print(samples)

            i = i + 1
            print('******************************SDR Receiver Running Loop**************************')
        self.sdr.close()
        self.finished.emit()


class MessengerWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, connected=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load configuration file config.yml
        with open("config.yml", "r") as ymlfile:
            self.cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        self.setupUi(self)

        self.pushButton.pressed.connect(self.button_send)
        self.pushButton_2.pressed.connect(self.connect)

    def button_send(self):
        text = self.textEdit.toPlainText()
        self.worker2.text = text
        self.send_message(text)
        self.textEdit.setText('')
        self.textEdit.repaint()

    def send_message(self, text):
        self.show_text('Me: ' + text)

    def connect(self):
        if self.pushButton_2.text() == 'Connect':
            self.pushButton_2.setText('Disconnect')
            ####################################################################
            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = SDR_receiver()
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            # begin our worker object's loop when the thread starts running
            self.thread.started.connect(self.worker.work)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # self.worker.progress.connect(self.reportProgress)
            # Step 6: Start the thread
            self.thread.start()

            #####################################################################
            # Step 2: Create a QThread object
            self.thread2 = QThread()
            # Step 3: Create a worker object
            self.worker2 = Tag_transmitter()
            # Step 4: Move worker to the thread
            self.worker2.moveToThread(self.thread2)
            # Step 5: Connect signals and slots
            # begin our worker object's loop when the thread starts running
            self.thread2.started.connect(self.worker2.work)
            self.worker2.finished.connect(self.thread2.quit)
            self.worker2.finished.connect(self.worker2.deleteLater)
            self.thread2.finished.connect(self.thread2.deleteLater)
            # self.worker.progress.connect(self.reportProgress)
            # Step 6: Start the thread
            self.thread2.start()

        elif self.pushButton_2.text() == 'Disconnect':
            self.pushButton_2.setText('Connect')
            # Stop the thread if pressed the Disconnect Button
            self.worker.working = False
            self.worker2.working = False

        # if self.pushButton_2.pressed:
        # input_text = query.lower()
        # self.textEdit.append(input_text)

    def show_text(self, text):
        self.textBrowser.append(text)
        self.textBrowser.repaint()

    def print_message(self, message):
        username = message['username']
        message_time = message['time']
        text = message['text']

        dt = datetime.fromtimestamp(message_time)
        dt_beauty = dt.strftime('%H:%M:%S')

        self.show_text(f'{dt_beauty} {username}\n{text}\n\n')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MessengerWindow()
    window.show()
    sys.exit(app.exec_())
