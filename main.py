#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:09:50 2022

@author: snr
"""
import os
import numpy as np
import scipy.signal as sg
from datetime import datetime
from visualizer import InfoWindow
from lsl_inlet import LSLInlet
from scipy.io import savemat
import pyqtgraph as pg
import PyQt6.QtWidgets as QtWidgets
import PyQt6.QtCore as QtCore
#import QtGui
import sys
import time

class Listener():
    def __init__(self):

        timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
        self.results_path = 'results/{}_{}/'.format('online_experiment', timestamp_str)
        os.makedirs(self.results_path)

        seq = ['Idling']

        self.exp_settings = {
            'exp_name': 'alpha',
            'lsl_stream_name': 'Mitsar',
            'blocks': {
                'Prepare': {'duration': 5, 'id': -1, 'message': '+'},
                'Idling': {'duration': 120, 'id': -2, 'message': 'Rest'},
                'Rest': {'duration': 15, 'id': 0, 'message': 'Rest'},
                'FB': {'duration': 100, 'id': 1, 'message': 'FeedBack'}},
            'sequence': seq,
            'seed': 1,

            #максимальная буферизация принятых данных через lsl
            'max_buflen': 1,  # in seconds!!!!
            #максимальное число принятых семплов в чанке, после которых появляется возможность
            #считать данные
            'max_chunklen': 1,  # in number of samples!!!!

            'visualization_window': 500,

            #какой канал использовать
            'channels_subset': 'P3-A1',
            'band': [9.0, 13.0],
            'results_path': self.results_path}

        #в онлайн версии определяется по мета данным потока
        self.srate = 500
        self.exp_settings['srate'] = self.srate
        self.time = np.arange(self.exp_settings['visualization_window'])/self.srate  # for visualization

        # connect to LSL stream

        self.inlet = LSLInlet(self.exp_settings)
        self.inlet.srate = 500
        self.xml_info = self.inlet.info_as_xml()
        self.channel_names = self.inlet.get_channels_labels()
        self.ch_idx = np.squeeze(np.where(self.channel_names ==
                                          np.array(self.exp_settings['channels_subset']))[0])
        self.n_channels = len(self.channel_names)

        block_durations = [self.exp_settings['blocks'][block_name]['duration']
                           for block_name in self.exp_settings['sequence']]
        n_seconds = sum(block_durations)

        self.buffer = np.empty((n_seconds * self.srate + 100 * self.srate, self.n_channels))
        self.buffer_stims = np.empty(n_seconds * self.srate + 100 * self.srate)

        self.n_samples_received = 0
        self.n_samples_received_in_block = 0
        self.block_idx = 0
        self.block_name = self.exp_settings['sequence'][0]
        self.current_block = self.exp_settings['blocks'][self.block_name]
        self.n_samples = self.srate * self.current_block['duration']
        self.block_id = self.current_block['id']



    def relay(self):

        self.receive_data_and_process()


    def receive_data_and_process(self):
        #print(f"{self.n_samples_received_in_block=}")
        if self.n_samples_received_in_block >= self.n_samples:
            self.block_idx += 1
            if self.block_idx >= len(self.exp_settings['sequence']):
                self.save_and_finish()
                self.inlet.disconnect()
                sys.exit()
                return

            self.block_name = self.exp_settings['sequence'][self.block_idx]
            self.current_block = self.exp_settings['blocks'][self.block_name]
            self.n_samples = self.srate * self.current_block['duration']
            self.n_samples_received_in_block = 0
            self.block_id = self.current_block['id']
            self.info_window.message = self.current_block['message']
            self.info_window.setText(listener.info_window.message)


        chunk, t_stamp = self.inlet.get_next_chunk()
        #print(f"{chunk=}")
        if chunk is not None:
            n_samples_in_chunk = len(chunk)
            self.buffer[self.n_samples_received:self.n_samples_received + n_samples_in_chunk, :] = chunk
            self.buffer_stims[self.n_samples_received:self.n_samples_received + n_samples_in_chunk] =\
                self.block_id
            self.n_samples_received_in_block += n_samples_in_chunk
            self.n_samples_received += n_samples_in_chunk

            if self.n_samples_received - self.exp_settings['visualization_window'] >= 0:
                self.graph.clear()
                self.graph.addLegend()
                self.graph.plot(self.time * 1000, np.squeeze(self.buffer[self.n_samples_received -
                       self.exp_settings['visualization_window']:self.n_samples_received,self.ch_idx]),
                       pen=pg.mkPen('w', width=2), name='eeg signal')

    def save_and_finish(self):
        # save recorded data
        print('here')
        recorded_data = self.buffer[:self.n_samples_received]
        recorded_stims = self.buffer_stims[:self.n_samples_received]
        savemat(self.results_path + 'data.mat', {'eeg': recorded_data, 'stim': recorded_stims,
                                                 'xml_info': self.xml_info})










if __name__ == "__main__":

    listener = Listener()

    app = QtWidgets.QApplication([])

    listener.info_window = InfoWindow()
    listener.info_window.setText(listener.current_block['message'])

    listener.plot_window =pg.GraphicsLayoutWidget()
    #надпись над графиком
    listener.txt = listener.plot_window.addLabel(0, 0)
    listener.txt.setText('any label')

    #график
    listener.graph = listener.plot_window.addPlot(1, 0)
    listener.graph.setLabel('left', "amplitude, uV")
    listener.graph.setLabel('bottom', "times, ms")  # , units='s')
    listener.plot_window.setGeometry(listener.info_window.top,
                        listener.info_window.left, listener.info_window.width, listener.info_window.height)
    listener.plot_window.ci.layout.setColumnMaximumWidth(0, listener.info_window.width)
    listener.plot_window.ci.layout.setRowMaximumHeight(0, listener.info_window.height)
    listener.plot_window.show()
    listener.graph.show()



    timer = QtCore.QTimer()
    timer.timeout.connect(listener.relay)
    timer.start(0)
    sys.exit(app.exec())





