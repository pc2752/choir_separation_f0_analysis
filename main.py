import models
import tensorflow as tf
import argparse
import os, sys
import config

def train(_):
    model = models.DeepSal()
    model.train()

def eval(file_name):
    model = models.DeepSal()
    scores = model.test_file(file_name)
    print("Evaluated file {}".format(file_name))
    for key, value in scores.items():
        print('{} : {}'.format(key, value))

def validate(file_name):
    model = models.DeepSal()
    scores = model.eval_all(file_name)
    import pdb;pdb.set_trace()

def eval_wav_file(file_name, save_path):
    model = models.DeepSal()
    scores = model.test_wav_file(file_name, save_path)

def eval_wav_folder(folder_name, save_path):
    model = models.DeepSal()
    scores = model.test_wav_folder(folder_name, save_path)

if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
        print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file" % sys.argv[0])
        print("%s -v or --v or -val or --val <filename> to calculate metrics for entire dataset and save to given filename" % sys.argv[0])
        print("%s -w or --w or -wavfile or --wavfile <filename> <save_path> to evaluate wavefile and save CSV" % sys.argv[0])
        print("%s -wf or --wf or -wavfolder or --wavolder <foldername> <save_path> to evaluate all wavefiles in the folder and save CSV" % sys.argv[0])
    else:
        if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
            print("Training")
            tf.app.run(main=train)
        elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
            if len(sys.argv)<3:
                print("Please give a file to evaluate")
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.hdf5'):
                    file_name = file_name+'.hdf5'
                if not file_name in os.listdir(config.feats_dir):
                    print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                else:
                    eval(file_name)
        elif sys.argv[1] == '-v' or sys.argv[1] == '--v' or sys.argv[1] == '--val' or sys.argv[1] == '-val':
            print("Evaluating entire validation set")
            if len(sys.argv)<3:
                print("Please give a file to evaluate")
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.csv'):
                    file_name = file_name+'.csv'
                validate(file_name)

        elif sys.argv[1] == '-w' or sys.argv[1] == '--w' or sys.argv[1] == '--wavfile' or sys.argv[1] == '-wavfile':
            print("Evaluating wave file")
            if len(sys.argv)<4:
                print("Please give a file and an output folder to evaluate")
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.wav'):
                    file_name = file_name+'.wav'
                save_path = sys.argv[3]
                eval_wav_file(file_name, save_path)

        elif sys.argv[1] == '-wf' or sys.argv[1] == '--wf' or sys.argv[1] == '--wavfolder' or sys.argv[1] == '-wavfolder':
            print("Evaluating entire validation set")
            if len(sys.argv)<4:
                print("Please give a file and an output folder to evaluate")
            else:
                folder_name = sys.argv[2]
                save_path = sys.argv[3]
                eval_wav_folder(folder_name, save_path)