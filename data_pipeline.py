import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils
import librosa
from scipy.ndimage import filters


def process_file(file_name):

    feat_file = h5py.File(config.feats_dir + file_name)

    atb = feat_file['atb'][()]

    atb = atb[:,1:]

    hcqt = feat_file['voc_hcqt'][()]

    feat_file.close()

    # import pdb;pdb.set_trace()

    return atb, np.array(hcqt)

def data_gen(mode = 'Train', sec_mode = 0):

    if mode == 'Train' :
        num_batches = config.batches_per_epoch_train
        file_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and not x.__contains__('1')]
    else:
        file_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.__contains__('1')]
        num_batches = config.batches_per_epoch_val


    max_files_to_process = int(config.batch_size / config.samples_per_file)



    for k in range(num_batches):

        out_hcqt = []
        out_atb = []

        for i in range(max_files_to_process):

            voc_index = np.random.randint(0, len(file_list))
            voc_file = file_list[voc_index]
            # atb, hcqt = process_file(voc_file)
            feat_file = h5py.File(config.feats_dir + voc_file)

            atb = feat_file['atb']

            atb = atb[:, 1:]

            hcqt = feat_file['voc_hcqt']

            for j in range(config.samples_per_file):
                voc_idx = np.random.randint(0, len(hcqt) - config.max_phr_len)
                out_hcqt.append(hcqt[voc_idx:voc_idx + config.max_phr_len])
                out_atb.append(atb[voc_idx:voc_idx + config.max_phr_len])

            feat_file.close()

            out_hcqt = np.array(out_hcqt)
            out_hcqt = np.swapaxes(out_hcqt, 2, 3)
            out_atb = np.array(out_atb)


            yield out_hcqt, out_atb






def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus')  and not x.startswith('nus_KENN') ]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        voc_stft = voc_file['voc_stft']

        feats = np.array(voc_file['feats'])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_voc[i]:
                max_voc[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_voc[i]:
                min_voc[i] = mini_voc_stft[i]

        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i]   

    for voc_to_open in back_list:

        voc_file = h5py.File(config.backing_dir+voc_to_open, "r")

        voc_stft = voc_file["back_stft"]

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_mix[i]:
                max_mix[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_mix[i]:
                min_mix[i] = mini_voc_stft[i]

    hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   
    hdf5_file.create_dataset("voc_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("voc_stft_minimus", [513], np.float32)   
    hdf5_file.create_dataset("back_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("back_stft_minimus", [513], np.float32)   

    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat
    hdf5_file["voc_stft_maximus"][:] = max_voc
    hdf5_file["voc_stft_minimus"][:] = min_voc
    hdf5_file["back_stft_maximus"][:] = max_mix
    hdf5_file["back_stft_minimus"][:] = min_mix

    # import pdb;pdb.set_trace()

    hdf5_file.close()


def get_stats_phonems():

    phon=collections.Counter([])

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
        pho_target = np.array(voc_file["phonemes"])
        phon += collections.Counter(pho_target)
    phonemas_weights = np.zeros(41)
    for pho in phon:
        phonemas_weights[pho] = phon[pho]

    phonemas_above_threshold = [config.phonemas[x[0]] for x in np.argwhere(phonemas_weights>70000)]

    pho_order = phonemas_weights.argsort()

    # phonemas_weights = 1.0/phonemas_weights
    # phonemas_weights = phonemas_weights/sum(phonemas_weights)
    import pdb;pdb.set_trace()


def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen('Train', sec_mode = 0)
    while True :
        start_time = time.time()
        ins, outs = next(gen)
        print(time.time()-start_time)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()