# from __future__ import division
import numpy as np
import librosa
import os
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

import h5py

import config

import sig_process
import utils
from acoufe import pitch

from scipy.ndimage import filters
import itertools
import time
# import essentia_backend as es

def process_f0(f0, f_bins, n_freqs):
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))

    haha = np.digitize(f0, f_bins) - 1

    idx2 = haha < n_freqs

    haha = haha[idx2]

    freqz[range(len(haha)), haha] = 1

    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T

    min_target = np.min(atb[range(len(haha)), haha])

    atb = atb / min_target

    # import pdb;pdb.set_trace()

    atb[atb > 1] = 1

    return atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins

def prep_deepsalience():
    x = [0,2,3,4]

    combos = [p for p in itertools.product(x, repeat=4)]

    combos = combos[1:]

    combos_2 = [p for p in itertools.product([0, 1], repeat=4)]

    combos = combos + combos_2[1:]

    songs = next(os.walk(config.wav_dir))[1]
    freq_grid = librosa.cqt_frequencies(config.cqt_bins, config.fmin, config.bins_per_octave)


    f_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    n_freqs = len(freq_grid)


    for song in songs:
        print ("Processing song %s" % song)
        song_dir = config.wav_dir+song+'/IndividualVoices/'
        singers = [x for x in os.listdir(song_dir) if x.endswith('.wav') and not x.startswith('.')]

        song_name = [x.split('_')[0] for x in singers][0]

        count = 0

        for combo in combos:
            # combo = [1,2,3,4]
            combo_str = str(combo[0]) + str(combo[1]) + str(combo[2]) + str(combo[3])

            if os.path.isfile(config.feats_dir+song_name+'_'+combo_str+'.hdf5'):
                if combo[0]!=0:
                    start_time = time.time()
                    audio_sop, fs = librosa.core.load(os.path.join(song_dir,song_name+'_soprano_'+str(combo[0])+'.wav'), sr = config.fs)
                    # print(time.time()-start_time)

                    start_time = time.time()
                    f0_sop = pitch.extract_f0_sac(audio_sop, fs, config.hoptime).reshape(-1,1)
                    # print(time.time()-start_time)
                    f0 = np.zeros((len(f0_sop),4))
                    f0[:,0] = f0_sop[:,0]
                    atb_sop = process_f0(f0_sop, f_bins, n_freqs)
                    audio = audio_sop
                    atb = atb_sop
                if combo[1]!=0:
                    # start_time = time.time()
                    audio_alt, fs = librosa.core.load(os.path.join(song_dir,song_name + '_alto_' + str(combo[1]) + '.wav'), sr=config.fs)
                    f0_alt = pitch.extract_f0_sac(audio_alt, fs, config.hoptime).reshape(-1,1)
                    atb_alt = process_f0(f0_alt, f_bins, n_freqs)
                    if combo[0] != 0:
                        audio = audio[:audio_alt.shape[0]] + audio_alt[:audio.shape[0]]
                        atb = atb[:atb_alt.shape[0]] + atb_alt[:atb.shape[0]]
                    else:
                        f0 = np.zeros((len(f0_alt),4))
                        audio = audio_alt
                        atb = atb_alt
                    f0[:f0_alt.shape[0],1] =f0_alt[:f0.shape[0],0]
                    # print(time.time()-start_time)
                if combo[2]!=0:
                    audio_bas, fs = librosa.core.load(os.path.join(song_dir, song_name + '_bass_' + str(combo[2]) + '.wav'), sr=config.fs)
                    f0_bas = pitch.extract_f0_sac(audio_bas, fs, config.hoptime).reshape(-1,1)
                    atb_bas = process_f0(f0_bas, f_bins, n_freqs)

                    if combo[0] == 0 and combo[1] ==0 :
                        audio = audio_bas
                        f0 = np.zeros((len(f0_bas),4))
                        atb = atb_bas
                    else:
                        audio = audio[:audio_bas.shape[0]] + audio_bas[:audio.shape[0]]
                        atb = atb[:atb_bas.shape[0]] + atb_bas[:atb.shape[0]]
                    f0[:f0_bas.shape[0],2] =f0_bas[:f0.shape[0],0]
                if combo[3]!=0:
                    audio_ten, fs = librosa.core.load(os.path.join(song_dir, song_name + '_tenor_' + str(combo[3]) + '.wav'), sr=config.fs)
                    f0_ten = pitch.extract_f0_sac(audio_ten, fs, config.hoptime).reshape(-1,1)
                    atb_ten = process_f0(f0_ten, f_bins, n_freqs)
                    if combo[0]== 0 and combo[1]== 0 and combo[2] == 0:
                        audio = audio_ten
                        f0 = np.zeros((len(f0_ten),4))
                        atb = atb_ten
                    else:
                        audio = audio[:audio_ten.shape[0]] + audio_ten[:audio.shape[0]]
                        atb = atb[:atb_ten.shape[0]] + atb_ten[:atb.shape[0]]
                    f0[:f0_ten.shape[0],3] =f0_ten[:f0.shape[0],0]

                num_sources = 4 - combo.count(0)
                audio = audio/num_sources
                # import pdb;pdb.set_trace()

                voc_hcqt = sig_process.get_feats(audio)

                # import pdb;pdb.set_trace()

                voc_hcqt, atb = utils.match_time([voc_hcqt,atb])

                # zeros = 1 - (f0==0).astype(int)



                assert voc_hcqt.shape[0] == f0.shape[0]




                hdf5_file = h5py.File(config.feats_dir+song_name+'_'+combo_str+'.hdf5', mode='a')

                # hdf5_file.create_dataset("zeros", zeros.shape, int)

                # hdf5_file.create_dataset("voc_cqt", voc_cqt.shape, np.complex64)

                if "voc_hcqt" in hdf5_file.keys():
                    del hdf5_file["voc_hcqt"]

                hdf5_file.create_dataset("voc_hcqt", voc_hcqt.shape, np.float64)

                # hdf5_file.create_dataset("f0", f0.shape, np.float64)
                if "atb" in hdf5_file.keys():
                    del hdf5_file["atb"]

                hdf5_file.create_dataset("atb", atb.shape, np.float64)



                # hdf5_file["zeros"][:,:] = zeros

                # hdf5_file["voc_cqt"][:,:] = voc_cqt

                hdf5_file["voc_hcqt"][:,:] = voc_hcqt

                hdf5_file["atb"][:, :] = atb

                # hdf5_file["f0"][:, :] = f0


                hdf5_file.close()



            count+=1

            utils.progress(count,len(combos))

def prep_voc_sep():

    freq_grid = librosa.cqt_frequencies(config.cqt_bins, config.fmin, config.bins_per_octave)


    f_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    n_freqs = len(freq_grid)
    songs = next(os.walk(config.wav_dir))[1]
    for song in songs:
        print ("Processing song %s" % song)
        song_dir = config.wav_dir+song+'/IndividualVoices/'
        song_names = [x for x in os.listdir(song_dir) if x.endswith('.wav') and not x.startswith('.') and not x.endswith('-24b.wav')]
        count = 0
        for song_name in song_names:

            

            audio, fs = librosa.core.load(os.path.join(song_dir,song_name), sr = config.fs)
            audio = np.array(audio)
            feats = sig_process.stft_to_feats(audio.astype('double'), fs, mode=config.comp_mode)
            voc_stft, voc_cqt, voc_hcqt = sig_process.get_feats(audio)
            f0 = pitch.extract_f0_sac(audio, fs, config.hoptime).reshape(-1,1)
            atb = process_f0(f0, f_bins, n_freqs)
            assert voc_stft.shape[0] == feats.shape[0]

            hdf5_file = h5py.File(config.voc_feats_dir+song_name+'.hdf5', mode='w')

            hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.complex64)

            hdf5_file.create_dataset("voc_cqt", voc_cqt.shape, np.complex64)

            hdf5_file.create_dataset("voc_hcqt", voc_hcqt.shape, np.float64)

            hdf5_file.create_dataset("voc_feats", feats.shape, np.float64)   

            hdf5_file.create_dataset("atb", atb.shape, np.float64) 

            hdf5_file["voc_stft"][:,:] = voc_stft

            hdf5_file["voc_cqt"][:,:] = voc_cqt

            hdf5_file["voc_hcqt"][:,:] = voc_hcqt

            hdf5_file["voc_feats"][:, :] = feats

            hdf5_file["atb"][:, :] = atb
            
            hdf5_file.close()


            count+=1

            utils.progress(count,len(song_names))


if __name__ == '__main__':
    #  prep_voc_sep()
    prep_deepsalience()
