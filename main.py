import models
import tensorflow as tf
import argparse
import os, sys
import config
import utils
import numpy as np
import mir_eval

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

def eval_helena(path_ori, path_est, save_path):
	songs = next(os.walk(path_ori))[1]
	scores = {}

	county = 0

		
	for song in songs[0:1]:
		song_folder_ori = os.path.join(path_ori, song)
		song_folder_est = os.path.join(path_est, song)

		quartets = [x for x in os.listdir(song_folder_ori) if x.endswith('.f0') and not x.startswith('.')]

		count = 0

		for quartet in quartets:

			ref_time_ori, ref_freqs_ori = utils.read_f0_file(os.path.join(song_folder_ori,quartet))

			max_index = np.argmax(ref_time_ori)

			ref_time_ori = ref_time_ori[:max_index]

			ref_freqs_ori = ref_freqs_ori[:max_index]

			ref_freqs_ori = utils.remove_zeros(ref_time_ori, ref_freqs_ori)

			ests = utils.csv_to_list(os.path.join(song_folder_est,quartet[:-3]+'.csv'))

			ref_time_est = [float(x[0].split('\t')[0]) for x in ests]

			ref_time_est = np.array(ref_time_est)

			ref_freqs_est = [np.array([float(y) for y in  x[0].split('\t')[1:]]) if len(x[0].split('\t'))>1 else np.array([]) for x in ests]

			file_score = mir_eval.multipitch.evaluate(ref_time_ori, ref_freqs_ori, ref_time_est, ref_freqs_est)
			if county == 0:
				for key, value in file_score.items():
					scores[key] = [value]
					scores['file_name'] = [song+quartet]
			else:
				for key, value in file_score.items():
					scores[key].append(value)
					scores['file_name'].append(song+quartet)
			count+=1
			county+=1

			utils.progress(count, len(quartets), suffix='evaluation done')
		import pdb;pdb.set_trace()
	# utils.save_scores_mir_eval(scores, save_path)
	# import pdb;pdb.set_trace()


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
			print("Evaluating entire validation set and saving to folder")
			if len(sys.argv)<4:
				print("Please give a file and an output folder to evaluate")
			else:
				folder_name = sys.argv[2]
				save_path = sys.argv[3]
				eval_wav_folder(folder_name, save_path)

		elif sys.argv[1] == '-he' or sys.argv[1] == '--he':
			print("Evaluating, please wait.")
			if len(sys.argv)<5:
				print("Please give path for orignal csvs, estimated csvs and a filename to save the results")
			else:
				path_ori= sys.argv[2]
				path_est = sys.argv[3]
				save_path = sys.argv[4]
				eval_helena(path_ori, path_est, save_path)

