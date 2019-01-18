import models
import tensorflow as tf
import argparse
import os, sys
import config

def train(_):
    model = models.DeepSal()
    model.train()

def synth(file_name):
    model = models.DeepSal()
    scores = model.test_file(file_name)
    print("Evaluated file {}".format(file_name))
    for key, value in scores.items():
        print('{} : {}'.format(key, value))

if __name__ == '__main__':
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        print("Training")
        tf.app.run(main=train)
    elif sys.argv[1] == '-e' or sys.argv[1] == '--eval' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
        if len(sys.argv)<3:
            print("Please give a file to evaluate")
        else:
            file_name = sys.argv[2]
            if not file_name.endswith('.hdf5'):
                file_name = file_name+'.hdf5'
            if not file_name in os.listdir(config.feats_dir):
                print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
            else:
                synth(file_name)
    elif sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --train or -t or --t or - train to train the model"%sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file"%sys.argv[0])