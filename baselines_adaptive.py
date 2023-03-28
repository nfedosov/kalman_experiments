# пишем данные 5 минут
# после сбора останавливаем запись
# применяем ICA/SCP
# визуализируем компоненты
# ищем моторную
# удаляем глаза и все ненужные
# сохраняем матрицу - идёт как параметр в фильтр
# смотрим на запись, вырезаем мышцы, делаем огибающую и выделяем пороги (желательно автоматически)



import mne
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
from lsl_inlet import LSLInlet
import cv2
import pickle
import scipy.signal as sn

timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
results_path = 'results/{}_{}/'.format('baseline_experiment', timestamp_str)
os.makedirs(results_path)


#parameters to change
seq = ['Rest', 'Move']*20
np.random.shuffle(seq)
seq2 = list()
seq2.append('Ready')
for i in range(len(seq)):
    seq2.append('Prepare')
    seq2.append(seq[i])



exp_settings = {
            'exp_name': 'baseline',
            'lsl_stream_name': 'NVX136_Data',
            'blocks': {
                'Prepare': {'duration': 1, 'id': -2, 'message': '+'},
                'Ready': {'duration': 10, 'id': -10, 'message': 'Rest'},
                'Rest': {'duration': 5, 'id': 0, 'message': 'Rest'},
                'Move': {'duration': 5, 'id': 1, 'message': 'Move'}},
            'sequence': seq2,
            'seed': 1,

            #максимальная буферизация принятых данных через lsl
            'max_buflen': 5,  # in seconds!!!!

            #максимальное число принятых семплов в чанке, после которых появляется возможность
            #считать данные
            'max_chunklen': 1,  # in number of samples!!!!


            ##какой канал использовать
            #'channels_subset': 'P3-A1',
            #'bands': [[9.0, 13.0],[18.0,25.0]],
            'results_path': results_path}





inlet = LSLInlet(exp_settings)
inlet.srate = inlet.get_frequency()
xml_info = inlet.info_as_xml()
channel_names = inlet.get_channels_labels()
print(channel_names)
exp_settings['channel_names'] = channel_names
#ch_idx = np.squeeze(np.where(channel_names == np.array(exp_settings['channels_subset']))[0])
n_channels = len(channel_names)


srate = int(round(inlet.srate)) #Hz
exp_settings['srate'] = srate

#max buffer length
total_samples = 0

for block_name in exp_settings['sequence']:
    current_block = exp_settings['blocks'][block_name]
    total_samples += round(srate * current_block['duration'])

data = np.zeros((total_samples+round(exp_settings['max_buflen']*srate),n_channels))
print(data.shape)
stims =  np.zeros((total_samples+round(exp_settings['max_buflen']*srate),1)).astype(int)


n_samples_received = 0
n_samples_received_in_block = 0


#buffer = np.empty((n_seconds * self.srate + 100 * self.srate, self.n_channels))
#buffer_stims = np.empty(n_seconds * self.srate + 100 * self.srate)

block_idx = 0
block_name = exp_settings['sequence'][0]
current_block = exp_settings['blocks'][block_name]
n_samples = srate * current_block['duration']
block_id = current_block['id']


cv2.namedWindow('stimwin', cv2.WINDOW_NORMAL)
cv2.resizeWindow('stimwin', 1950, 1250)
cv2.moveWindow('stimwin', -1950, 200)
 
move = cv2.imread("move.jpg")
rest = cv2.imread("rest.jpg")
prepare = cv2.imread("prepare.jpg")
ready = cv2.imread("ready.jpg")

if block_name == 'Rest':
    print("Rest")
    cv2.imshow('stimwin', rest)
    cv2.waitKey(1)  # ??????
    # cv2.imshow('action', ...)

if block_name == 'Move':
    print("Move")
    cv2.imshow('stimwin', move)
    cv2.waitKey(1)  # ??????
if block_name == 'Prepare':
    print("Prepare")
    cv2.imshow('stimwin', prepare)
    cv2.waitKey(1)  # ??????

if block_name == 'Ready':
    print("Ready")
    cv2.imshow('stimwin', ready)
    cv2.waitKey(1)  # ??????

while (1):
    if n_samples_received_in_block >= n_samples:
        dif =  n_samples_received_in_block - n_samples
        
        
        block_idx += 1
        
        
        if block_idx >= len(exp_settings['sequence']):
            #save_and_finish()
            inlet.disconnect()

            break

        block_name = exp_settings['sequence'][block_idx]
        current_block = exp_settings['blocks'][block_name]
        n_samples = srate * current_block['duration']
        n_samples_received_in_block = dif
        block_id = current_block['id']
        if dif>0:
            stims[n_samples_received-dif:n_samples_received] = block_id
        
        

        #message shpw
        #maybe not openCV?????
        if block_name == 'Rest':
            print("Rest")
            cv2.imshow('stimwin', rest)
            cv2.waitKey(1) #??????
            #cv2.imshow('action', ...)
            
        if block_name == 'Move':
            print("Move")
            cv2.imshow('stimwin', move)
            cv2.waitKey(1) #??????
        if block_name == 'Prepare':
            print("Prepare")
            cv2.imshow('stimwin', prepare)
            cv2.waitKey(1) #??????
            
        if block_name == 'Ready':
            print("Ready")
            cv2.imshow('stimwin', ready)
            cv2.waitKey(1) #??????
            
    


    chunk, t_stamp = inlet.get_next_chunk()
    # print(f"{chunk=}")
    if chunk is not None:
        n_samples_in_chunk = len(chunk)
        data[n_samples_received:n_samples_received + n_samples_in_chunk, :] = chunk
        stims[n_samples_received:n_samples_received + n_samples_in_chunk] = block_id
        n_samples_received_in_block += n_samples_in_chunk
        n_samples_received += n_samples_in_chunk




data = data[:total_samples]
stims = stims[:total_samples,0]


file = open(results_path + 'data.pickle', "wb")
pickle.dump({'eeg': data, 'stim': stims,
        'xml_info': xml_info, 'exp_settings': exp_settings}, file = file)
file.close()




print('Finished')
