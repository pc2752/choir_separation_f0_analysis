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

    feat_file = h5py.File(config.feats_dir + file_name, 'r')

    atb = feat_file['atb'][()]

    atb = atb[:,1:]

    hcqt = feat_file['voc_hcqt'][()]

    feat_file.close()

    # import pdb;pdb.set_trace()

    return atb, np.array(hcqt)

def data_gen(mode = 'Train', sec_mode = 0):
    stat_file = h5py.File('./stats.hdf5', mode='r')

    max_f0 = stat_file["f0_maximus"][()]
    max_cqt = stat_file["cqt_maximus"][()]
    stat_file.close()

    if mode == 'Train' :
        num_batches = config.batches_per_epoch_train
        file_list = config.train_list
        # import pdb;pdb.set_trace()
    else:
        file_list = config.val_list
        num_batches = config.batches_per_epoch_val


    max_files_to_process = int(config.batch_size / config.samples_per_file)



    for k in range(num_batches):

        out_hcqt = []
        out_atb = []
        out_f0 = []
        out_zeros = []


        for i in range(max_files_to_process):

            voc_index = np.random.randint(0, len(file_list))
            voc_file = file_list[voc_index]
            # atb, hcqt = process_file(voc_file)
            feat_file = h5py.File(config.feats_dir_2 + voc_file, 'r')

            f0 = feat_file['f0'][()]/max_f0

            atb = feat_file['atb']

            atb = atb[:, 1:]

            atb[:, 0:4] = 0

            atb = np.clip(atb, 0.0, 1.0)

            # atb = filters.gaussian_filter1d(atb.T, 0.5, axis=0, mode='constant').T



            # atb = np.clip(atb, 0.0, 1.0)

            hcqt = feat_file['voc_hcqt'][()]

            zeros = feat_file['zeros']

            for j in range(config.samples_per_file):
                voc_idx = np.random.randint(0, len(hcqt) - config.max_phr_len)
                out_hcqt.append(hcqt[voc_idx:voc_idx + config.max_phr_len])
                out_atb.append(atb[voc_idx:voc_idx + config.max_phr_len])
                out_f0.append(f0[voc_idx:voc_idx + config.max_phr_len])
                out_zeros.append(zeros[voc_idx:voc_idx + config.max_phr_len])

            feat_file.close()

            out_hcqt = np.array(out_hcqt)
            out_hcqt = np.swapaxes(out_hcqt, 2, 3)
            if config.add_noise:
                out_cqt = np.random.rand(out_cqt.shape) * config.noise_threshold + out_cqt
            out_f0 = np.array(out_f0)

            out_zeros = np.array(out_zeros)

            out_atb = np.array(out_atb)


            yield out_hcqt, out_f0, out_zeros, out_atb


def sep_gen(mode = 'Train', sec_mode = 0):

    stat_file = h5py.File('./stats.hdf5', mode='r')

    max_feat = stat_file["feats_maximus"][()]
    min_feat = stat_file["feats_minimus"][()]


    stat_file.close()

    if mode == 'Train' :
        num_batches = config.batches_per_epoch_train
        file_list = config.train_list
        # import pdb;pdb.set_trace()
    else:
        file_list = config.val_list
        num_batches = config.batches_per_epoch_val


    max_files_to_process = int(config.batch_size / config.samples_per_file)


    for k in range(num_batches):

        voice = np.random.randint(0,4)

        if voice == 0:
            shortlist = [x for x in file_list if x[-9]!='0']
            voc_part = '_soprano_'
            voc_num = -9

        elif voice == 1:
            shortlist = [x for x in file_list if x[-8]!='0']
            voc_part = '_alto_'
            voc_num = -8

        elif voice == 2:
            shortlist = [x for x in file_list if x[-7]!='0']
            voc_part = '_bass_' 
            voc_num = -7

        elif voice == 3:
            shortlist = [x for x in file_list if x[-6]!='0']
            voc_part = '_tenor_'
            voc_num = -6

        out_hcqt = []
        out_atb = []

        out_feats = []


        for i in range(max_files_to_process):

            voc_index = np.random.randint(0, len(shortlist))
            voc_file = shortlist[voc_index]

            song_name = voc_file.split('_')[0]

            voc_track = voc_file[voc_num]

            feat_file = h5py.File(config.feats_dir + voc_file, 'r')

            cqt = feat_file['voc_cqt']

            voc_feat_file = h5py.File(config.voc_feats_dir + song_name+voc_part+voc_track+'.wav.hdf5', 'r')

            voc_feats = voc_feat_file["voc_feats"][()]

            voc_feats[np.argwhere(np.isnan(voc_feats))] = 0.0

            atb = voc_feat_file['atb']

            atb = atb[:, 1:]

            atb[:, 0:4] = 0

            atb = np.clip(atb, 0.0, 1.0)

            max_len = min(len(voc_feats), len(cqt))

            voc_feats = voc_feats[:max_len]

            cqt = cqt[:max_len]

            atb = atb[:max_len]
            
            for j in range(config.samples_per_file):
                voc_idx = np.random.randint(0, len(hcqt) - config.max_phr_len)
                out_hcqt.append(hcqt[voc_idx:voc_idx + config.max_phr_len])
                out_atb.append(atb[voc_idx:voc_idx + config.max_phr_len])
                out_feats.append(voc_feats[voc_idx:voc_idx + config.max_phr_len])

            feat_file.close()

            out_hcqt = np.array(out_hcqt)
            # out_hcqt = np.swapaxes(out_hcqt, 2, 3)
            if config.add_noise:
                out_hcqt = np.random.rand(out_hcqt.shape) * config.noise_threshold + out_hcqt
            out_atb = np.array(out_atb)
            out_feats = np.array(out_feats)

            out_feats = (out_feats-min_feat)/(max_feat-min_feat)

            out_feats = np.clip(out_feats[:, :, :-2],0.0 , 0.1)


            yield abs(out_hcqt), out_atb, out_feats




def get_stats():
    voc_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5')]


    max_feat_f0 = np.zeros(4)
    min_feat_f0 = np.ones(4)*1000
    max_feat_cqt = 0
    min_feat_cqt = 1 

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.feats_dir+voc_to_open, "r")

        # import pdb;pdb.set_trace()

        f0 = voc_file["f0"][()]
        cqt = abs(voc_file["voc_cqt"][()])




        maxi_voc_feat = np.array(f0).max(axis=0)
        maxi_cqt = cqt.max()
        mini_cqt = cqt.min()
        if maxi_cqt > max_feat_cqt:
            max_feat_cqt = maxi_cqt
        if mini_cqt < min_feat_cqt:
            min_feat_cqt = mini_cqt
        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat_f0[i]:
                max_feat_f0[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(f0).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat_f0[i]:
                min_feat_f0[i] = mini_voc_feat[i]   

    # import pdb;pdb.set_trace()


    hdf5_file = h5py.File('./stats.hdf5', mode='w')

    hdf5_file.create_dataset("f0_maximus", [4], np.float32) 
    hdf5_file.create_dataset("cqt_maximus", [1], np.float32)   


    hdf5_file["f0_maximus"][:] = max_feat_f0
    hdf5_file["cqt_maximus"][:] = max_feat_cqt




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
        ins, outs, feats, booboo = next(gen)
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
