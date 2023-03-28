from pylsl.pylsl import StreamOutlet, StreamInfo
import numpy as np
import time
import matplotlib.pyplot as plt

import mne
raw= mne.io.read_raw_brainvision('C:/Users/CBI/Documents/NeoRec/Records/mu_test/test1/NeoRec_2021-04-15_14-08-30.vhdr')
#raw.plot()
#plt.show()




chNames = ['C3','Cz', 'F3','Fz','Fc1','Fc5','Cp5']
chname2 = {'C3': 'C3','Cz': 'Cz','F3': 'F3','Fz': 'Fz','Fc1': 'FC1','Fc5': 'FC5','Cp5': 'CP5'}
raw= raw[chNames][0]
print(raw.shape)
chCount = len(chNames)

class BCIStreamOutlet(StreamOutlet):
    def __init__(self, name = 'NVX136_Data', fs = 500):
        # create stream info
        info = StreamInfo(name=name, type='BCI', channel_count=chCount, nominal_srate=fs,
                          channel_format='float32', source_id='myuid34234')

        # set channels labels (in accordance with XDF format, see also code.google.com/p/xdf)
        chns = info.desc().append_child("channels")
        for chname in chNames:
            ch = chns.append_child("channel")
            ch.append_child_value("label", chname2[chname])

        # init stream
        super(BCIStreamOutlet, self).__init__(info,chunk_size = 1)


fs = 500
outlet = BCIStreamOutlet(fs = fs)

start_time = time.time()
model_time = 0.0
Ns = 1
count = 0;
while(1):
    cur_time = time.time()
    if cur_time-start_time > model_time+Ns/fs:

        x = raw[:,count].tolist()#np.random.randn(chCount*Ns,).tolist()
        outlet.push_chunk(x)
        model_time += Ns/fs
        count += 1

    if model_time > 1000.0:
        del outlet
        break