import os,re
import collections
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import pyworld as pw
from reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp, mgc_to_mfsc, mfsc_to_mgc
from acoufe import pitch

import config
import utils

import librosa




def get_hcqt(audio):

    cqt_list = []
    shapes = []
    for h in config.harmonics:
        
        cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin*float(h), bins_per_octave = config.bins_per_octave)
        cqt_list.append(cqt.T)

    cqt_list = utils.match_time(cqt_list)
    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt




def stft_to_feats(vocals, fs, mode=config.comp_mode):
    feats=pw.wav2world(vocals,fs,frame_period=config.hoptime*1000)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10*np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    f0 = pitch.extract_f0_sac(vocals, fs, config.hoptime)

    y=69+12*np.log2(f0/440)
    # import pdb;pdb.set_trace()
    # y = hertz_to_new_base(f0)
    nans, x= utils.nan_helper(y)

    naners=np.isinf(y)

    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)

    if mode == 'mfsc':
        harmy=sp_to_mfsc(harm,60,0.45)
        apy=sp_to_mfsc(ap,4,0.45)
    elif mode == 'mgc':
        harmy=sp_to_mgc(harm,60,0.45)
        apy=sp_to_mgc(ap,4,0.45)
    out_feats=np.concatenate((harmy,apy,y.reshape((-1,2))),axis=1) 

    return out_feats

def get_feats(audio):



    stft = librosa.core.stft(audio, n_fft = config.nfft, hop_length = config.hopsize, window = config.window).T

    # voc_stft_mag = 2 * abs(voc_stft)/np.sum(config.window)

    # voc_stft_phase = np.angle(voc_stft)

    cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin, bins_per_octave = config.bins_per_octave).T

    hcqt = get_hcqt(audio)

    hcqt = np.swapaxes(hcqt, 0,1)

    return stft, cqt, hcqt