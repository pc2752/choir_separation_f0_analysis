import numpy as np
import os
import re
import mir_eval
import utils
import config

def cents_to_hz(freq):
	if freq == 0:
		return 
	else:
		return 10*np.power(2, float(freq/1200))

op_dir = './res_12/'
gt_dir = '/home/helenacuesta/12singers/cents/'


files = [x for x in os.listdir(op_dir) if x.endswith('.txt') and not x.startswith('.')]

count = 0

for file_name in files:

	file_op = open(os.path.join(op_dir,file_name))
	filer_gt = open(os.path.join(gt_dir, file_name))
	op = file_op.readlines()
	file_op.close()
	op = np.array([[float(y) for y in x.replace('[','').replace(']','').replace('(','').replace(')','').replace(',','').split()] for x in op])

	lens = np.array([len(x) for x in op])

	time_1 = [x[0] for x in op]
	freqs = []

	for x, y in zip(op, lens):
		if y == 9:
			freqs.append(np.array([cents_to_hz(x[-2]), cents_to_hz(x[-4]),cents_to_hz(x[-6]), cents_to_hz(x[-8])]))
		elif y == 7:
			freqs.append(np.array([cents_to_hz(x[-2]), cents_to_hz(x[-4]),cents_to_hz(x[-6])]))
		elif y == 5:
			freqs.append(np.array([cents_to_hz(x[-2]), cents_to_hz(x[-4])]))
		elif y == 3:
			freqs.append(np.array(list(filter(None,[cents_to_hz(x[-2])]))))
		else:
			print("len error")
	# import pdb;pdb.set_trace()


	gt = filer_gt.readlines()
	filer_gt.close()
	gt =  np.array([[float(y) for y in x.split()] for x in gt])



	idx = []
	for i in range(len(gt)):
		if (gt[i][1]==gt[i][2] and gt[i][1]!=0) or (gt[i][3]==gt[i][4] and gt[i][3]!=0) or (gt[i][5]==gt[i][6] and gt[i][5]!=0) or (gt[i][7]==gt[i][8] and gt[i][7]!=0):
			idx.append(i)

	freqs_gt = [[cents_to_hz(x[1]), cents_to_hz(x[3]), cents_to_hz(x[5]),cents_to_hz(x[7])] for x in gt]

	freqs_gt = [np.array(list(filter(None,x))) for x in freqs_gt]

	op = op[:len(gt)]
	time_1 = time_1[:len(gt)]
	freqs = freqs[:len(gt)]

	time_2 = [x[0] for x in gt]

	time_3 = [x for i,x in enumerate(time_2) if i not in idx]

	freqs_gt = [x for i,x in enumerate(freqs_gt) if i not in idx]

	freqs = [x for i,x in enumerate(freqs) if i not in idx]

	time_1 = [x for i,x in enumerate(time_1) if i not in idx]

	time = np.linspace(0, config.hoptime * len(freqs), len(freqs))

	scores = mir_eval.multipitch.evaluate(time, np.array(freqs_gt), time, np.array(freqs))

	if count == 0:
		with open('./eval_12.txt','w') as f:
			f.write('File Name, ')
			for x in scores.keys():
				f.write(str(x)+', ')
			f.write('\n')
			f.write(file_name+', ')
			for x in scores.keys():
				f.write(str(scores[x])+', ')
			f.write('\n')
	else:
		with open('./eval_12.txt','a') as f:
			f.write(file_name+', ')
			for x in scores.keys():
				f.write(str(scores[x])+', ')
			f.write('\n')
		# import pdb;pdb.set_trace()
	count+=1
	utils.progress(count, len(files))




# lens = [x for i,x in enumerate(lens) if i not in idx]


# op = [x for i,x in enumerate(op) if i not in idx]

# gt = [x for i,x in enumerate(gt) if i not in idx]

# op_4 = op[lens == 9]

# gt_4 = gt[lens ==9]

# dif_mean_sop_2 = np.array([(x[1] - y[-2])**2 for x,y in zip(gt_4,op_4)])

# dif_mean_sop_4 = np.array([[(x[1] - y[-2])**2,x[1],y[-2]] for x,y in zip(gt_4,op_4) if not x[1] == x[2]])

import pdb;pdb.set_trace()
