# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:16:12 2023

@author: Fedosov
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal as sn
from filters import get_filtered

plt.close('all')

results_path= 'results/baseline_experiment_03-10_20-58-05/'


postfix = ['','2','3','4']#['','2','3','4']
labels = np.array([])
eyes  =np.array([])
alpha = np.array([])


for fix in postfix:
    file = open(results_path + 'experiment_results'+fix+'.pickle', "rb")
    dictionary = pickle.load(file)
    file.close()
    
    labels = np.concatenate([labels,dictionary['labels']])
    data = dictionary['data']
    eyes = np.concatenate([eyes,data@eye_filter])
    alpha = np.concatenate([alpha,data@ica_filter])
    
from scipy.signal import butter, lfilter

b,a =butter(1,[0.8,20.0], btype = 'bandpass', fs = 500)
b50,a50 = butter(1,[48.0,52.0],btype = 'bandstop',fs = 500)
eyes = lfilter(b,a,eyes)
eyes = lfilter(b50,a50,eyes)

b,a =butter(1,[2.0,30.0], btype = 'bandpass', fs = 500)
b50,a50 = butter(1,[48.0,52.0],btype = 'bandstop',fs = 500)
alpha = lfilter(b,a,alpha)
alpha = lfilter(b50,a50,alpha)

'''
file = open(results_path + 'model.pickle', "rb")
dictionary = pickle.load(file)
file.close()


kf_alpha = dictionary['kf_alpha']
cfir_beta = dictionary['cfir_beta']


alpha_ica_filter = dictionary['alpha_ica_filter']
beta_ica_filter = dictionary['beta_ica_filter']
b = dictionary['b']
a = dictionary['a']
b50 = dictionary['b50']
a50 = dictionary['a50']
smoother_alpha_kf = dictionary['smoother_alpha_kf']
smoother_beta_cfir = dictionary['smoother_beta_cfir']

hqa_kf =dictionary['hqa_kf']


alpha = alpha_ica_filter@data.T
beta = beta_ica_filter@data.T

alpha = sn.lfilter(b,a,alpha)
alpha = sn.lfilter(b50,a50,alpha)
alpha = get_filtered(kf_alpha.KF,alpha)

alpha_envelope = np.sqrt(alpha[:,0]**2+alpha[:,1]**2)
alpha_envelope = smoother_alpha_kf.apply(alpha_envelope)


plt.figure()
plt.plot(alpha_envelope)
plt.plot(np.arange(len(alpha_envelope)),np.tile(hqa_kf,len(alpha_envelope)))
labels[labels != 1] =0
labels = labels/1e1

plt.plot(labels/1000)

#beta = sn.lfilter(b,a,beta)
#beta = sn.lfilter(b50,a50,beta)
#beta = get_filtered(kf_beta.KF,beta)



plt.show()

'''

bad_idx = np.abs(eyes)>3e-5

bad_idx_final = np.zeros(bad_idx.shape, dtype = bool)
for i in range(-100,100):
    bad_idx_shifted = np.roll(bad_idx,i)
    bad_idx_final = (bad_idx_shifted+bad_idx)>0


# 8 
reacts = [[],[],[],[]]
labels = labels.astype('int')

count = 0

while count < len(labels):
    if (labels[count] != 0) and (labels[count] != 111):
    #if (labels[count] != 0) and (bad_idx_final[count] != True) and (labels[count] != 111):
        fix_count = count
        while  labels[count] != 111:
            count += 1
            
        if (labels[fix_count]) == 5:
            idx = 3
        if (labels[fix_count]) == 6:
            idx = 4
        if (labels[fix_count]) == 1:
            idx = 1
        if (labels[fix_count]) == 2:
            idx = 2
        reacts[idx-1].append(count-fix_count)
        
    count += 1
        
        
reacts = np.array(reacts)

#mean_reacts = np.mean(reacts,axis = 1)    


names = ['high_alpha_kf','low_alpha_kf',
         'high_alpha_cfir','low_alph_cfir']
react_range = np.linspace(np.min(np.min(reacts))-1,250,20)
#for i in range(reacts.shape[0]):
#    plt.figure()
#    plt.hist(reacts[i],react_range,)
#    plt.title(names[i])


plt.figure()
plt.hist(reacts[0],react_range)
plt.title(names[0])


plt.hist(reacts[1],react_range, alpha = 0.5)
plt.title(names[1])

plt.figure()
plt.hist(reacts[2],react_range)
plt.title(names[2])


plt.hist(reacts[3],react_range, alpha = 0.5)
plt.title(names[3])




from scipy.stats import mannwhitneyu


labels[labels == 111] = 0
for i in range(2):
    
    
    
    react_ha = np.array(reacts[i*2])
    react_ha = react_ha[react_ha<250.0]
    print(np.mean(react_ha))
    
    react_la = np.array(reacts[i*2+1])
    react_la = react_la[react_la<250.0]
    print(np.mean(react_la))
    
    plt.figure()
    plt.plot(alpha[5000:])
    
    
    #plt.plot()
    
    plt.plot(labels[5000:]*0.05*np.max(eyes))
    plt.plot(np.ones(len(labels[5000:]))*0.05*np.max(eyes)*1,'--')
    plt.plot(np.ones(len(labels[5000:]))*0.05*np.max(eyes)*2,'--')
    plt.plot(np.ones(len(labels[5000:]))*0.05*np.max(eyes)*5,'--')
    plt.plot(np.ones(len(labels[5000:]))*0.05*np.max(eyes)*6,'--')
    
    
    plt.plot(eyes[5000:],alpha = 0.5)
    
    u_statistic, p_value = mannwhitneyu(react_ha, react_la)
    print(p_value)
















