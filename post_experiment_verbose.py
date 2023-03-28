# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:16:12 2023

@author: Fedosov
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal as sn

from kalman_experiments import lfilter_kf_alpha_beta

plt.close('all')

results_path= 'results/baseline_experiment_03-10_20-58-05/'

#03-14_21-27-38
#03-10_20-58-05


file = open(results_path + 'experiment_results.pickle', "rb")
dictionary = pickle.load(file)
file.close()

labels = dictionary['labels']
data = dictionary['data']


file = open(results_path + 'model.pickle', "rb")
dictionary = pickle.load(file)
file.close()



hqa_kf =dictionary['hqa_kf']
lqa_kf =dictionary['lqa_kf']
hqb_kf =dictionary['hqb_kf']
lqb_kf =dictionary['lqb_kf']

hqa_cfir =dictionary['hqa_cfir']
lqa_cfir =dictionary['lqa_cfir']
hqb_cfir =dictionary['hqb_cfir']
lqb_cfir =dictionary['lqb_cfir']


kf = dictionary['kf']


cfir_alpha = dictionary['cfir_alpha']
cfir_beta = dictionary['cfir_beta']
b = dictionary['b']
a = dictionary['a']
b50 = dictionary['b50']
a50 = dictionary['a50']
b100 = dictionary['b100']
a100 = dictionary['a100']
b150 = dictionary['b150']
a150 = dictionary['a150']
b200 = dictionary['b200']
a200 = dictionary['a200']


ica_filter = dictionary['ica_filter']


smoother_alpha_cfir = dictionary['smoother_alpha_cfir']
smoother_beta_cfir = dictionary['smoother_beta_cfir']


n_samples_received = 0

#data = np.zeros((800*10*srate,n_channels))

z = np.zeros(len(b)-1)
z50 = np.zeros(len(b50)-1)
z100 = np.zeros(len(b100)-1)
z150 = np.zeros(len(b150)-1)
z200 = np.zeros(len(b200)-1)


nT = 100000


cum_env_kf_alpha = np.zeros(nT)
cum_env_kf_beta = np.zeros(nT)
cum_env_cfir_alpha = np.zeros(nT)
cum_env_cfir_beta = np.zeros(nT)

cum_kf_alpha = np.zeros((nT,2))
cum_kf_beta = np.zeros((nT,2))
cum_cfir_alpha = np.zeros((nT,2))
cum_cfir_beta = np.zeros((nT,2))


chunk_cum = np.zeros((nT))

n_samples_in_chunk = 7
for i in range(nT//7):
    

        chunk  = data[i*7:(i+1)*7,:]
           
        chunk = ica_filter@chunk.T
          
        chunk, z  = sn.lfilter(b,a,chunk, zi = z)
        chunk, z50  = sn.lfilter(b50,a50,chunk, zi = z50)
        chunk, z100  = sn.lfilter(b100,a100,chunk, zi = z100) 
        chunk, z150  = sn.lfilter(b150,a150,chunk, zi = z150) 
        chunk, z200  = sn.lfilter(b200,a200,chunk, zi = z200)
        
        chunk_cum[i*7:(i+1)*7] = chunk

        
     
        envelope_kf_alpha,envelope_kf_beta, kf_alpha,kf_beta = lfilter_kf_alpha_beta(kf, chunk)
           
        filtered_cfir_alpha = cfir_alpha.apply(chunk)
        filtered_cfir_beta = cfir_beta.apply(chunk) 

                  
              
        envelope_cfir_alpha =np.abs(filtered_cfir_alpha)          
        envelope_cfir_beta =np.abs(filtered_cfir_beta)
        
        cfir_alpha_rh = filtered_cfir_alpha
        cfir_beta_rh = filtered_cfir_beta
           
        envelope_cfir_alpha = smoother_alpha_cfir.apply(envelope_cfir_alpha)
        envelope_cfir_beta = smoother_beta_cfir.apply(envelope_cfir_beta)
           
        cum_env_kf_alpha[i*7:(i+1)*7] = envelope_kf_alpha
        cum_env_kf_beta[i*7:(i+1)*7] = envelope_kf_beta
        cum_kf_alpha[i*7:(i+1)*7] = kf_alpha
        cum_kf_beta[i*7:(i+1)*7] = kf_beta
        
        cum_env_cfir_alpha[i*7:(i+1)*7] = envelope_cfir_alpha
        cum_env_cfir_beta[i*7:(i+1)*7] = envelope_cfir_beta
        cum_cfir_alpha[i*7:(i+1)*7,0] = np.real(cfir_alpha_rh)
        cum_cfir_beta[i*7:(i+1)*7,0] = np.real(cfir_beta_rh)
        cum_cfir_alpha[i*7:(i+1)*7,1] = np.imag(cfir_alpha_rh)
        cum_cfir_beta[i*7:(i+1)*7,1] = np.imag(cfir_beta_rh)
           
    
           
    
plt.figure()
plt.plot(cum_env_kf_alpha[2000:])
plt.plot(cum_kf_alpha[2000:,0])
#plt.plot(cum_kf_alpha[2000:,1])
plt.plot(chunk_cum[2000:])

hqa_line = np.ones(len(cum_kf_alpha[2000:,0]))*hqa_kf
lqa_line = np.ones(len(cum_kf_alpha[2000:,0]))*lqa_kf
plt.plot(hqa_line,'--')
plt.plot(lqa_line,'--')
plt.ylim(np.min(cum_kf_alpha[2000:,0]*10),np.max(cum_kf_alpha[2000:,0]*10))

#plt.plot(cum_env_cfir_alpha[2000:]*5)



f,pxx_alpha = sn.welch(cum_kf_alpha[2000:,0], nperseg = 500,fs = 500)

f,pxx_beta = sn.welch(cum_kf_beta[2000:,0], nperseg = 500,fs = 500)


plt.figure()
plt.plot(f,np.log10(pxx_alpha))
plt.plot(f,np.log10(pxx_beta))


plt.figure()
plt.plot(cum_env_cfir_alpha[2000:])
plt.plot(cum_cfir_alpha[2000:,0])
#plt.plot(cum_kf_alpha[2000:,1])
plt.plot(chunk_cum[2000:])

hqa_line = np.ones(len(cum_cfir_alpha[2000:,0]))*hqa_cfir
lqa_line = np.ones(len(cum_cfir_alpha[2000:,0]))*lqa_cfir
plt.plot(hqa_line,'--')
plt.plot(lqa_line,'--')
plt.ylim(np.min(cum_cfir_alpha[2000:,0]*10),np.max(cum_cfir_alpha[2000:,0]*10))

plt.figure()
plt.imshow(corr_mem)




plt.figure()
plt.plot(cum_env_cfir_alpha[2000:])
plt.plot(cum_cfir_alpha[2000:,0])
plt.plot(cum_env_kf_alpha[2000:])
plt.plot(cum_kf_alpha[2000:,0])







'''



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

