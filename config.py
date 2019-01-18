import numpy as np
# import tensorflow as tf


wav_dir = '../datasets/ChoralSingingDataset/'


feats_dir = './feats/'


#FFT Parameters
fs = 44100.0
nfft = 1024
hopsize = 512
hoptime = hopsize/fs
window = np.hanning(nfft)


#CQT Parameters
fmin = 32.70
bins_per_octave = 60
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves
harmonics = [0.5, 1, 2, 3, 4, 5]

comp_mode = 'mfsc'

# Model parameters
batch_size = 5
max_phr_len = 50
max_models_to_keep = 10
log_dir = './log/'
init_lr = 0.0001
batches_per_epoch_train = 100
batches_per_epoch_val = 10
samples_per_file = 5
num_epochs = 10000
print_every = 1
validate_every = 10
save_every = 10