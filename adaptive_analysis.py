

import mne
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import cv2
import scipy.signal as sn
import pickle
from functools import partial
#from filters import ExponentialSmoother,CFIRBand,grid_optimization, gen_ar_noise_coefficients, theor_psd_ar, PerturbedP1DMatsudaKF, MatsudaParams, get_filtered
from kalman_experiments import CFIRBand, ExponentialSmoother, theor_psd_ar, AlphaBetaKF, apply_kf_alpha_beta, phase_envelope_loss
from kalman_experiments import complex2mat, vec2complex
from kalman_experiments import PerturbedPKF, apply_kalman_interval_smoother
from kalman_experiments import OneDimKF, PerturbedP1DMatsudaKF, apply_kf
from kalman_experiments import ArNoiseModel,MatsudaParams,SingleRhythmModel,gen_ar_noise_coefficients

from kalman_experiments import fit_kf_parameters, normalize_measurement_dimensions
from kalman_experiments import estimate_sigmas, get_psd_val_from_est, optimize_kf_params, ideal_envelope

from kalman_experiments import to_db, grid_optimizer


plt.close('all')

np.random.seed(0)


pathdir = 'results/baseline_experiment_03-10_20-58-05/'

#alpha,beta




file = open(pathdir+'data.pickle', "rb")
container =  pickle.load(file)
file.close()

exp_settings = container['exp_settings']
srate = exp_settings['srate']
data = container['eeg']
stims = container['stim']
channel_names = exp_settings['channel_names']
for i in range(9):
    channel_names[i] = channel_names[i].upper()
print(channel_names)
n_channels = len(channel_names)




#transfrom to mne format

# for test

####


####


info = mne.create_info(ch_names=channel_names, sfreq = srate, ch_types = 'eeg',)
raw =  mne.io.RawArray(data.T, info)

print(data.shape)
print(info)

### ASSUME THAT MOVE AND REST DURATIONS ARE EQUAL
duration = exp_settings['blocks']['Move']['duration']
move_id = exp_settings['blocks']['Move']['id']
rest_id = exp_settings['blocks']['Rest']['id']
prepare_id = exp_settings['blocks']['Prepare']['id']
ready_id = exp_settings['blocks']['Ready']['id']



start_times = (np.where(np.isin(stims[1:]-stims[:-1], [move_id -prepare_id,rest_id-prepare_id]))[0]+1)/srate


description = []
for st in start_times:
    if stims[int(round((st*srate)))] == move_id:
        description.append('Move')

    if stims[int(round((st*srate)))] == rest_id:
        description.append('Rest')

    #if stims[int(round((st*srate)))] == prepare_id:
    #    description.append('Prepare')

#description =  stims[(start_times*srate).astype(int)].astype(str)
print(description, start_times, duration)




annotations = mne.Annotations(start_times, duration, description)
raw.set_annotations(annotations)


montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)



raw.plot(scalings = dict(eeg=1e0))
plt.show()
raw_copy = raw.copy()

#HERE MANUALLY MARK BAD SEGMENTS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




#raw.plot_psd()

#input('Press <Space> if you have marked bad segments...')

bad_segments = []

bad_segments.append(np.arange(5*srate))
for annot in raw._annotations:
    if annot['description'] == 'BAD_':
        #print('HERE')
        bad_segments = bad_segments + np.arange(int(round(annot['onset']*srate)),int(round((annot['onset']+annot['duration'])*srate))).tolist()


last_idx = int(round(raw[:][1][-1]*srate))

bad_segments.append(np.arange(last_idx-5*srate,last_idx+1))
bad_segments =np.concatenate(bad_segments)
good_idx = np.setdiff1d(np.arange(last_idx+1),bad_segments)



raw.notch_filter([50.0,100.0,150.0,200.0])
raw.filter(2.0,30.0)

ica = mne.preprocessing.ICA()
ica.fit(raw, start = int(5*srate), stop = int(last_idx-5*srate))

ics = ica.get_sources(raw)
ica.plot_sources(raw)

events = mne.events_from_annotations(raw)

ics_move = mne.Epochs(ics,events[0], tmin = 0, tmax = duration, event_id = events[1]['Move'], baseline = None)
ics_rest = mne.Epochs(ics,events[0], tmin = 0, tmax = duration, event_id = events[1]['Rest'],baseline = None)

rel_alphas =np.zeros(n_channels)
rel_betas = np.zeros(n_channels)
central_freq_alphas= np.zeros(n_channels)
central_freq_betas= np.zeros(n_channels)

fig, axs = plt.subplots(3, 3)
for i in range(n_channels):

    ics_move.plot_psd(ax = axs[i//3, i%3], picks = [i,], color = 'red', spatial_colors = False,fmin = 2.0,fmax = 30)
    ics_rest.plot_psd(ax = axs[i//3, i%3], picks = [i,], color = 'blue', spatial_colors = False,fmin  = 2.0,fmax = 30)
    axs[i//3, i%3].set_title(str(i))
    
    psd_alpha_rest = ics_rest.compute_psd(fmin = 9.0, fmax = 13.0,picks = [i,]).get_data(return_freqs = True)
    psd_beta_rest = ics_rest.compute_psd(fmin = 16.0, fmax = 26.0,picks = [i,]).get_data(return_freqs = True)
    
    psd_alpha_move = ics_move.compute_psd(fmin = 9.0, fmax = 13.0,picks = [i,]).get_data(return_freqs = True)
    psd_beta_move = ics_move.compute_psd(fmin = 16.0, fmax = 26.0,picks = [i,]).get_data(return_freqs = True)
    
    rel_alphas[i] = np.mean(psd_alpha_rest[0])/np.mean(psd_alpha_move[0])
    rel_betas[i] = np.mean(psd_beta_rest[0])/np.mean(psd_beta_move[0])
    
    central_freq_alphas[i]= psd_alpha_rest[1][np.argmax(np.mean(psd_alpha_rest[0],axis = 0))]
    central_freq_betas[i] = psd_beta_rest[1][np.argmax(np.mean(psd_beta_rest[0],axis = 0))]
    
    
    

    


ica.plot_components()


    
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#str_idx = input('write an integer idx of SMR component...\n')
alpha_idx = np.argmax(rel_alphas)
print('index of SMR component is ', alpha_idx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#central_alpha = float(input('write alpha central frequency...\n'))
#central_beta = float(input('write beta central frequency...\n'))
central_alpha = central_freq_alphas[alpha_idx]
central_beta = central_freq_betas[alpha_idx]

print('central alpha freq is ', central_alpha)
print('central beta freq is ', central_beta)




#????? можно сделать тоже вводимыми значениями
bands = [[central_alpha - 2.0, central_alpha + 2.0],[central_beta-3.0,central_beta+3.0]]




pca_M = ica.pca_components_

ica_M = ica.unmixing_matrix_

unmixing_M = ica_M@pca_M

ica_filter = unmixing_M[alpha_idx,:]
#beta_ica_filter = alpha_ica_filter.copy()

alpha_rhythm = ics[alpha_idx,:][0][0,:]
beta_rhythm = alpha_rhythm.copy()


#plt.figure()
#plt.plot(alpha_rhythm)
#plt.plot(ica_filter@raw[:][0]*0.2e6)


#bad_ica = [] # WRITE HERE BAD ICA numbers
#I = np.eye(n_channels)
#I[bad_ica,bad_ica] = 0
#filtering_matrix = np.linalg.inv(unmixing_M)@ I@ unmixing_M




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


f,pxx = sn.welch(ics_rest.get_data()[:,alpha_idx,:],axis = 1, fs = srate, nperseg = srate)
pxx = np.mean(pxx,axis = 0)
plt.figure()
plt.plot(f[:40],np.log10(pxx[:40]))

alpha_coef_range = np.linspace(0.2,2.0,10)
order = 30
a_arr = np.zeros((len(alpha_coef_range),order))
for i, alpha_coef in enumerate(alpha_coef_range):
    
    a_arr[i] = gen_ar_noise_coefficients(alpha=alpha_coef, order=30)


#DOes it eliminates
    
    ar_pxx = theor_psd_ar(f,1e-2,a_arr[i],srate)

    
    plt.plot(f[:40],np.log10(ar_pxx[:40]))
    
plt.legend(['alpha_component']+np.arange(len(alpha_coef_range)).astype('str').tolist())

plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#a_idx = int(input('Type the INT number of the line with closest slope\n'))
a_idx = 4
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




ar = a_arr[a_idx]
print(ar)
raw_realt = raw_copy[:][0]


# q_s, r_s - ????
freq_alpha = (bands[0][1]+bands[0][0])/2
freq_beta = (bands[1][1]+bands[1][0])/2
A_alpha = 0.992
A_beta = 0.98

b50,a50=sn.butter(2,[45.0,55.0], btype = 'bandstop',fs = srate)
b100,a100 = sn.butter(1,[96.0,104.0], btype = 'bandstop',fs = srate)
b150,a150 = sn.butter(1,[144.0,156.0], btype = 'bandstop',fs = srate)
b200,a200 = sn.butter(1,[192.0,208.0], btype = 'bandstop',fs = srate)
b,a = sn.butter(1,[1.0,100.0], btype = 'bandpass', fs =srate)

realt  = ica_filter@raw_realt
realt = ica_filter@raw_realt



realt = sn.lfilter(b50,a50,realt)
realt = sn.lfilter(b100,a100,realt)
realt = sn.lfilter(b150,a150,realt)
realt = sn.lfilter(b200,a200,realt)
realt = sn.lfilter(b,a,realt)






cfir_alpha = CFIRBand(bands[0], srate, n_taps = 50)
cfir_beta = CFIRBand(bands[1], srate, n_taps = 50)

mp_alpha = MatsudaParams(A=A_alpha, freq=freq_alpha, sr=srate)
mp_beta = MatsudaParams(A=A_beta, freq=freq_beta, sr=srate)


mk_model_alpha = SingleRhythmModel(mp_alpha, sigma=1)
mk_model_beta = SingleRhythmModel(mp_beta, sigma=1)


ff, psd = sn.welch(realt, fs=srate, nperseg=srate*4)

est_psd_func = partial(get_psd_val_from_est, freqs=ff, psd=psd)
ar_noise_model = ArNoiseModel(
    order=order, alpha=alpha_coef_range[a_idx], sr=srate, x0=np.random.randn(order), sigma=1
)
fit_freqs = [6, freq_alpha, 14, freq_beta, 40]

#q_s_alpha_2, q_s_beta_2, r_s_2 = estimate_sigmas(
#    [mk_model_alpha.psd_onesided, mk_model_beta.psd_onesided, ar_noise_model.psd_onesided],
#    est_psd_func,
#    fit_freqs)



#corr_mem, lat_mem = grid_optimizer(realt, good_idx, freq_alpha, freq_beta, ar, srate,log_num = 10, q_s_alpha_bounds = [2,-5],q_s_r_bounds = [2,-5], q_s_beta_rate = 10.0)
    


#res = optimize_kf_params(realt,srate,freq_alpha,freq_beta, good_idx, ar, bias = None)
#file = open(pathdir + 'res.pickle', "rb")
#dic = pickle.load(file)
#res = dic['res']


#q_s_alpha, q_s_beta, r_s = np.sqrt(q_s_alpha_2), np.sqrt(q_s_beta_2), np.sqrt(r_s_2)


mp_alpha = MatsudaParams(A=0.999, freq=freq_alpha, sr=srate)
mp_beta = MatsudaParams(A=0.999, freq=freq_beta, sr=srate)


q_s_beta = 1e-05*10.0#res[1]
r_s = 0.006#res[2]
q_s_alpha = 1e-05#res[0]

kf = AlphaBetaKF(mp_alpha=mp_alpha, q_s_alpha=q_s_alpha, mp_beta=mp_beta, q_s_beta=q_s_beta, r_s=r_s, psi=ar)


kf_states_alpha, kf_states_beta = apply_kf_alpha_beta(kf, signal=realt)
                      
#kf_alpha = PerturbedP1DMatsudaKF(MatsudaParams(A=A_alpha, freq = freq_alpha, sr = srate), q_s=1, psi=ar, r_s=1, lambda_=0)
#kf_beta = PerturbedP1DMatsudaKF(MatsudaParams(A=A_beta, freq = freq_beta, sr = srate), q_s=1, psi=ar, r_s=1, lambda_=0)


#kf = fit_kf_parameters(alpha_rhythm, kf_alpha, n_iter = 1)
#kf.KF.Q[0,0] = 0.01
#kf.KF.Q[1,1] = 0.01
#kf.KF.Q[2,2] = 10.0


#q_range = np.array([1.0,]) #np.logspace(-3.0, 1.0, num=4, base=10.0)
#r_range = np.logspace(-2.0, 4.0, num=9, base=10.0)#np.logspace(-1.0, 3.0, num=4, base=10.0)
#env,lat = grid_optimization(kf_alpha, alpha_realt.copy(), q_range = q_range, r_range = r_range, bad_idx = bad_segments)


alpha_envelope = np.abs(kf_states_alpha)[good_idx]
beta_envelope = np.abs(kf_states_beta)[good_idx]


gt_alpha_envelope = ideal_envelope(freq_alpha, srate, realt)[good_idx]
gt_beta_envelope = ideal_envelope(freq_beta, srate, realt)[good_idx]






# №№№№№№№№№№№№№№№№№№№№№№№№№№№
'''
import scipy.stats as stat
realt2 = stat.zscore(realt)
x, P = apply_kf(kf.KF,realt2)
x_n, P_n, J = apply_kalman_interval_smoother(kf.KF, x[-100:],P[-100:])

x_np = np.array(x_n)
gt_alpha_envelope_kf = np.linalg.norm(x_np[:,:2,0],axis = 1)
'''
# №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№






alpha_cc_cum = np.zeros(99)
for i,bias in enumerate(range(1,100)):


    gt_alpha_envelope_trim = gt_alpha_envelope[:-bias]
    gt_beta_envelope_trim = gt_beta_envelope[:-bias]
    
    gt_alpha_envelope_trim -= np.mean(gt_alpha_envelope)
    gt_beta_envelope_trim -= np.mean(gt_beta_envelope)

    
    
    alpha_envelope_trim = alpha_envelope[bias:]
    beta_envelope_trim = beta_envelope[bias:]


    alpha_envelope_trim -= np.mean(alpha_envelope_trim)
    beta_envelope_trim -= np.mean(beta_envelope_trim)
    


    alpha_cc_cum[i] = np.sum(alpha_envelope_trim*gt_alpha_envelope_trim)/(np.linalg.norm(alpha_envelope_trim)*np.linalg.norm(gt_alpha_envelope_trim))
    #beta_cc_cum[i] = np.sum(beta_envelope*gt_beta_envelope)/(np.linalg.norm(beta_envelope)*np.linalg.norm(gt_beta_envelope))


plt.figure()
plt.plot(np.arange(1,100)*1000/srate,alpha_cc_cum)


#env, lat = phase_envelope_loss(srate,central_alpha, np.abs(kf_states_alpha),realt,bad_idx = [])
#print('alpha env corr')
#print(np.round(env,2))
#print('alpha latency, ms')
#print(lat*1000/srate) #ms



#env, lat = phase_envelope_loss(srate,central_beta, np.abs(kf_states_beta),realt,bad_idx = [])
#print('beta env corr')
#print(np.round(env,2))
#print('beta latency, ms')
#print(lat*1000/srate) #ms




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
print(q_s_alpha)
print(q_s_beta)
print(r_s)

#q_idx = int(input('type idx of the optimal q (row):\n'))
#r_idx = int(input('type idx of the optimal r (column):\n'))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#kf_alpha.KF.Q[0][0] = q_range[q_idx]
#kf_alpha.KF.Q[1][1] = q_range[q_idx]
#kf_alpha.KF.Q[2][2] = r_range[r_idx]

# FOR BETA THE SAME LIKE ALPHA!!!
#kf_beta.KF.Q[0][0] = q_range[q_idx]
#kf_beta.KF.Q[1][1] = q_range[q_idx]
#kf_beta.KF.Q[2][2] = r_range[r_idx]




smoother_alpha_cfir = ExponentialSmoother(0.95)
smoother_beta_cfir = ExponentialSmoother(0.95)
     


#kf.q_s = 0.01
#kf.r_s = 10.0

# meas, KF, n_iter: int = 50, tol: float = 1e-3

#alpha_rhythm += np.sin(np.arange(alpha_rhythm.shape[0])*(freq_alpha*2*np.pi)/srate)*0.1






#pure_alpha_kf = get_filtered(kf_alpha.KF, alpha_realt)
alpha_envelope_kf = np.abs(kf_states_alpha)

beta_envelope_kf = np.abs(kf_states_beta)

#alpha_ica_filter@raw_realt
pure_alpha_cfir = cfir_alpha.apply(realt)
alpha_envelope_cfir = np.abs(pure_alpha_cfir)
alpha_envelope_cfir = smoother_alpha_cfir.apply(alpha_envelope_cfir)
#beta_ica_filter@raw_realt
pure_beta_cfir = cfir_beta.apply(realt)
beta_envelope_cfir = np.abs(pure_beta_cfir)
beta_envelope_cfir = smoother_beta_cfir.apply(beta_envelope_cfir)






#plt.figure()
#plt.plot(pure_alpha)
#plt.plot(alpha_rhythm)
#plt.plot(alpha_envelope)


from scipy.signal import welch

f,pxx = welch(np.real(kf_states_alpha)[good_idx],fs = srate,nperseg = srate)
plt.figure()
plt.plot(f[:100],np.log10(pxx[:100]))
#
#
# f,pxx = welch(pure_beta_kf[good_idx,0],fs = srate,nperseg = srate)
# plt.figure()
# plt.plot(f[:50],np.log10(pxx[:50]))


f,pxx = welch(np.imag(pure_alpha_cfir)[good_idx],fs = srate,nperseg = srate)
plt.figure()
plt.plot(f[:100],np.log10(pxx[:100]))

# f,pxx = welch(np.real(pure_beta_cfir)[good_idx],fs = srate,nperseg = srate)
# plt.figure()
# plt.plot(f[:50],np.log10(pxx[:50]))





#consider bad segments

plt.figure()
plt.hist(alpha_envelope_kf[good_idx],bins = 40, range = [np.min(alpha_envelope_kf[good_idx]),np.max(alpha_envelope_kf[good_idx])])

hq = 0.97
lq = 0.1

high_quantile_alpha_kf = np.quantile(alpha_envelope_kf[good_idx],hq)
low_quantile_alpha_kf = np.quantile(alpha_envelope_kf[good_idx],lq)
plt.figure()
plt.plot(realt, alpha = 0.5)
plt.plot(np.real(kf_states_alpha))
plt.plot(alpha_envelope_kf)
length = len(alpha_envelope_kf)
plt.plot(np.arange(length),[high_quantile_alpha_kf]*length)
plt.plot(np.arange(length),[low_quantile_alpha_kf]*length)


high_quantile_beta_kf = np.quantile(beta_envelope_kf[good_idx],hq)
low_quantile_beta_kf = np.quantile(beta_envelope_kf[good_idx],lq)
plt.figure()
plt.plot(realt, alpha = 0.5)
plt.plot(np.real(kf_states_beta))
plt.plot(beta_envelope_kf)
length = len(beta_envelope_kf)
plt.plot(np.arange(length),[high_quantile_beta_kf]*length)
plt.plot(np.arange(length),[low_quantile_beta_kf]*length)



high_quantile_alpha_cfir = np.quantile(alpha_envelope_cfir[good_idx],hq)
low_quantile_alpha_cfir = np.quantile(alpha_envelope_cfir[good_idx],lq)
plt.figure()
plt.plot(realt, alpha = 0.5)
plt.plot(np.real(pure_alpha_cfir))
plt.plot(alpha_envelope_cfir)
length = len(alpha_envelope_cfir)
plt.plot(np.arange(length),[high_quantile_alpha_cfir]*length)
plt.plot(np.arange(length),[low_quantile_alpha_cfir]*length)


high_quantile_beta_cfir = np.quantile(beta_envelope_cfir[good_idx],hq)
low_quantile_beta_cfir = np.quantile(beta_envelope_cfir[good_idx],lq)
plt.figure()
plt.plot(realt, alpha = 0.5)
plt.plot(np.real(pure_beta_cfir))
plt.plot(beta_envelope_cfir)
length = len(beta_envelope_cfir)
plt.plot(np.arange(length),[high_quantile_beta_cfir]*length)
plt.plot(np.arange(length),[low_quantile_beta_cfir]*length)








file = open(pathdir + 'model.pickle', "wb")
pickle.dump({'hqa_kf': high_quantile_alpha_kf,'lqa_kf': low_quantile_alpha_kf,
             'hqb_kf': high_quantile_beta_kf,'lqb_kf': low_quantile_beta_kf,
             'hqa_cfir': high_quantile_alpha_cfir,'lqa_cfir': low_quantile_alpha_cfir,
             'hqb_cfir': high_quantile_beta_cfir,'lqb_cfir': low_quantile_beta_cfir,
            'kf': kf,'cfir_alpha': cfir_alpha,'cfir_beta':cfir_beta, 'ica_filter': ica_filter,
           'b': b,'a': a, 'b50': b50,'a50': a50, 'b100':b100,'a100':a100,'b150':b150,'a150':a150,'b200':b200,'a200':a200,
            'smoother_alpha_cfir':smoother_alpha_cfir,'smoother_beta_cfir':smoother_beta_cfir}, file = file)
file.close()
#

plt.show()


#estimate envelope



# save filtering vector for the particular component, and freq filtration ranges, both for beta and alpha
#savemat(..., )



#CHOSING AND DELETING COMPONENT
#SAVING THE FILTERING MATRIX
#MIN MAX AND THRESHOLD ESTIMATION



#maybe not necessary, but the second ica or csp




#events = mne.events_from_annotations(annotations)

#events_id =
#epochs =mne.Epochs(raw,events, tmin = 0, tmax = dur



