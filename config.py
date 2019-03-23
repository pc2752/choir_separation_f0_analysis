import numpy as np
import os
# import tensorflow as tf


wav_dir = '../datasets/ChoralSingingDataset/'


feats_dir = './feats_full_res/'
voc_feats_dir = './voc_feats/'


#FFT Parameters
fs = 44100.0
nfft = 1024
hopsize = 512
hoptime = hopsize/fs
window = np.hanning(nfft)


#CQT Parameters
fmin = 32.70
bins_per_octave = 60
# bins_per_octave = 24
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves
harmonics = [0.5, 1, 2, 3, 4, 5]

comp_mode = 'mfsc'

feats_dir_2 = feats_dir

# Model parameters
batch_size = 5
max_phr_len = 50
max_models_to_keep = 10
log_dir = './log_res_full/'
init_lr = 0.0001
batches_per_epoch_train = 100
batches_per_epoch_val = 10
samples_per_file = 5
num_epochs = 2000
print_every = 1
validate_every = 10
save_every = 50
# train_list = [x for x in os.listdir(feats_dir_2) if x.endswith('.hdf5') and not x.__contains__('1') and not x.startswith('nino')]
# val_list = [x for x in os.listdir(feats_dir_2) if x.endswith('.hdf5') and x.__contains__('1') or x.startswith('nino')]

train_list = [x for x in os.listdir(feats_dir_2) if x.endswith('.hdf5')  and not x== 'nino_4424.hdf5 ']
val_list = ['nino_4424.hdf5']

# train_list = [x for x in os.listdir(feats_dir_2) if x.endswith('.hdf5')  and not x.startswith('nino')]
# val_list = [x for x in os.listdir(feats_dir_2) if x.endswith('.hdf5') and x.startswith('nino')]

add_noise = False
noise_threshold = 0.2
wavenet_filters = 128
wavenet_layers = 5

# Sep Model
log_dir_sep = './log_sep/'

# Phase 2
min_max = {'tenor': [65.40639132514966, 523.2511306011972], 'soprano': [108.0, 880.0], 'alto': [108, 880.0], 'bass': [55.0, 440.0]}
min_max_midi = {'tenor': ['C2', 'C5'], 'soprano': ['A2', 'A5'], 'alto': ['A2', 'A5'], 'bass': ['A1', 'A4']}

bins_per_octave_2 = 117

