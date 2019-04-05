import tensorflow as tf
from modules_tf import DeepSalience_1, DeepSalience_2, DeepSalience
import config
from data_pipeline import data_gen, sep_gen
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
import sig_process
import matplotlib.pyplot as plt
from scipy.ndimage import filters



class Model(object):
    def __init__(self):
        self.get_placeholders()
        self.model()
        stat_file = h5py.File('./stats.hdf5', mode='r')

        self.max_f0 = stat_file["f0_maximus"][()]
        self.max_cqt = stat_file["cqt_maximus"][()]
        stat_file.close()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec

    def validate_file_1(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """


        in_batches_hcqt, atb, nchunks_in = self.read_input_file(file_name)
        out_batches_atb = []
        for in_batch_hcqt in in_batches_hcqt:
            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
            out_atb = sess.run(self.outputs, feed_dict=feed_dict)
            out_batches_atb.append(out_atb)
        out_batches_atb = np.array(out_batches_atb)
        out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)
        out_batches_atb = out_batches_atb[:atb.shape[0]]

        baba = np.mean(np.equal(np.round(atb[atb>0]), np.round(out_batches_atb[atb>0])))

        atb = filters.gaussian_filter1d(atb.T, 0.5, axis=0, mode='constant').T

        #
        time_1, ori_freq = utils.process_output(atb)
        time_2, est_freq = utils.process_output(out_batches_atb)


        scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)

        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.cqt_bins, 6),name='input_placeholder')
        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.cqt_bins),name='output_placeholder')
        self.output_placeholder_1 = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 4),name='output_placeholder_1')
        self.output_placeholder_2 = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 4),name='output_placeholder_2')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)




        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.optimizer_1 = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.optimizer_2 = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_1 = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_2 = tf.Variable(0, name='global_step_2', trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_function = self.optimizer.minimize(self.loss, global_step = self.global_step_1)
            self.train_function_1 = self.optimizer.minimize(self.loss_1, global_step = self.global_step_1)
            self.train_function_2 = self.optimizer_2.minimize(self.nll, global_step = self.global_step_2)


    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """
        self.loss_summary = tf.summary.scalar('final_loss', self.loss)
        self.loss_1_summary = tf.summary.scalar('first_loss', self.loss_1)
        self.nll_summary = tf.summary.scalar('nll_loss', self.nll)
        self.rmse_summary = tf.summary.scalar('RMSE', self.rmse)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))


class DeepSal(Model):

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output_logits))
        self.loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder_1, logits = self.output_logits_1))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round( self.output_placeholder_1 ), tf.round(self.outputs_1)), tf.float32)*self.output_placeholder_1)
        self.nll = utils.nll_gaussian(self.output_mean*self.output_placeholder_1, self.output_std, self.output_placeholder_2)
        # self.rmse = tf.losses.mean_squared_error(utils.hz_to_cents(self.output_placeholder_2 * self.max_f0) , utils.hz_to_cents(self.output_mean*self.outputs_1*self.max_f0))
        self.rmse = tf.losses.mean_squared_error(self.output_placeholder_2 * self.max_f0 , self.output_mean*self.outputs_1*self.max_f0)


    def read_input_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        feat_file = h5py.File(config.feats_dir + file_name)
        atb = feat_file['atb'][()]

        atb = atb[:, 1:]

        hcqt = feat_file['voc_hcqt'][()]

        feat_file.close()

        in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*360))
        in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
                                                  6, 360)
        in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)
        return in_batches_hcqt, atb, nchunks_in

    def read_input_file_1(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        feat_file = h5py.File(config.feats_dir + file_name)
        atb = feat_file['f0'][()]

        # atb = atb[:, 1:]

        hcqt = abs(feat_file['voc_cqt'][()])

        feat_file.close()

        in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt)
        # in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*config.cqt_bins))
        # in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
        #                                           6, config.cqt_bins)
        # in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)
        return in_batches_hcqt, atb, nchunks_in

    def read_input_wav_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        audio, fs = librosa.core.load(file_name, sr=config.fs)
        hcqt = sig_process.get_hcqt(audio/4)

        hcqt = np.swapaxes(hcqt, 0, 1)

        in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*config.cqt_bins))
        in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
                                                  6, config.cqt_bins)
        in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)

        return in_batches_hcqt, nchunks_in, hcqt.shape[0]




    def test_file(self, file_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def test_wav_file(self, file_name, save_path):
        """
        Function to extract multi pitch from wav file.
        """

        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        in_batches_hcqt, nchunks_in, max_len = self.read_input_wav_file(file_name)
        out_batches_atb = []
        for in_batch_hcqt in in_batches_hcqt:
            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
            out_atb = sess.run(self.outputs, feed_dict=feed_dict)
            out_batches_atb.append(out_atb)
        out_batches_atb = np.array(out_batches_atb)
        out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)
        out_batches_atb = out_batches_atb[:max_len]
        # plt.imshow(out_batches_atb.T, origin = 'lower', aspect = 'auto')
        #
        # plt.show()
        # import pdb;pdb.set_trace()

        time_1, ori_freq = utils.process_output(out_batches_atb)
        utils.save_multif0_output(time_1, ori_freq, save_path)


    def test_wav_folder(self, folder_name, save_path):
        """
        Function to extract multi pitch from wav files in a folder
        """

        songs = next(os.walk(folder_name))[1]

        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        

        for song in songs:
            count = 0
            print ("Processing song %s" % song)
            file_list = [x for x in os.listdir(os.path.join(folder_name, song)) if x.endswith('.wav') and not x.startswith('.')]
            for file_name in file_list:
                in_batches_hcqt, nchunks_in, max_len = self.read_input_wav_file(os.path.join(folder_name, song, file_name))
                out_batches_atb = []
                for in_batch_hcqt in in_batches_hcqt:
                    feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
                    out_atb = sess.run(self.outputs, feed_dict=feed_dict)
                    out_batches_atb.append(out_atb)
                out_batches_atb = np.array(out_batches_atb)
                out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
                                 nchunks_in)
                out_batches_atb = out_batches_atb[:max_len]
                time_1, ori_freq = utils.process_output(out_batches_atb)
                utils.save_multif0_output(time_1, ori_freq, os.path.join(save_path,song,file_name[:-4]+'.csv'))
                count+=1
                utils.progress(count, len(file_list), suffix='evaluation done')

    def extract_f0_file(self, file_name, sess):
        in_batches_hcqt, atb, nchunks_in = self.read_input_file(file_name)
        out_batches_atb = []
        for in_batch_hcqt in in_batches_hcqt:
            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
            out_atb = sess.run(self.outputs, feed_dict=feed_dict)
            out_batches_atb.append(out_atb)
        out_batches_atb = np.array(out_batches_atb)
        out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)
        out_batches_atb = out_batches_atb[:atb.shape[0]]
        time_1, ori_freq = utils.process_output(atb)
        time_2, est_freq = utils.process_output(out_batches_atb)

        scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)
        return scores

    def extract_f0_file_1(self, file_name, sess):
        # if file_name in config.val_list:
        #     mode = "Val"
        # else:
        #     mode = "Train"
        # num_singers = file_name.count('1')
        # song_name = file_name.split('_')[0].capitalize()
        # voice = config.log_dir.split('_')[-1][:-1].capitalize()

        in_batches_hcqt, atb, nchunks_in = self.read_input_file(file_name)
        out_batches_mean = []
        out_batches_std = []
        for in_batch_hcqt in in_batches_hcqt:
            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
            out_mean, out_std = sess.run([self.output_mean, self.output_std], feed_dict=feed_dict)
            out_batches_mean.append(out_mean)
            out_batches_std.append(out_std)
        out_batches_mean = np.array(out_batches_mean)
        out_batches_mean = utils.overlapadd(out_batches_mean,nchunks_in)
        plt.plot(atb[:,0], label = 'GT')
        plt.plot(out_batches_mean[:,0]*self.max_f0[0], label = 'Pred')
        plt.legend()
        plt.show()
        import pdb;pdb.set_trace()
        out_batches_mean = utils.overlapadd(out_batches_mean.reshape(out_batches_mean.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)
        out_batches_mean = out_batches_atb[:mean.shape[0]]

        baba = np.mean(np.equal(np.round(atb[atb>0]), np.round(out_batches_atb[atb>0])))

        atb = filters.gaussian_filter1d(atb.T, 0.5, axis=0, mode='constant').T


        plt.figure(1)
        plt.suptitle("Note Probabilities for song {}, voice {}, with {} singers, from the {} set".format(song_name, voice,num_singers, mode) + "bin activation accuracy: {0:.0%}".format(baba), fontsize=10)
        ax1 = plt.subplot(211)

        plt.imshow(np.round(atb.T), origin = 'lower', aspect = 'auto')

        ax1.set_title("Ground Truth Note Probabilities (10 cents per bin)", fontsize=10)
        ax2 = plt.subplot(212, sharex = ax1, sharey=ax1)
        plt.imshow(np.round(out_batches_atb.T), origin='lower', aspect='auto')
        ax2.set_title("Output Note Probabilities (10 cents per bin)", fontsize=10)
        plt.show()

        cont = utils.query_yes_no("Do you want to see probability distribution per frame? Default No", default = "no")

        while cont:

            num_sings = int(input("How many distinct pitches per frame to plot. Default {}".format(num_singers)) or num_singers)


            index = np.random.choice(np.where(atb.sum(axis=1)==num_sings)[0])
            plt.figure(1)
            plt.suptitle("Probability Distribution For one of the Frames With {} Distinct Pitches in GT".format(num_singers))
            ax1 = plt.subplot(211)
            ax1.set_title("Ground Truth Probability Distribution Across Frame", fontsize=10)
            plt.plot(np.round(atb[index]))
            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
            plt.plot(np.round(out_batches_atb[index]))
            ax2.set_title("Output Probability Distribution Across Frame", fontsize=10)
            plt.show()
            cont = utils.query_yes_no("Do you want to see probability distribution per frame? Default No", default="no")

        #
        time_1, ori_freq = utils.process_output(atb)
        time_2, est_freq = utils.process_output(out_batches_atb)

        utils.save_multif0_output(time_1, ori_freq, './gt.csv')
        utils.save_multif0_output(time_2, est_freq, './op.csv')

        scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)
        return scores

        # import pdb;pdb.set_trace()

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            # if (epoch+1) % 10 == 0:
            #     import pdb;pdb.set_trace()
            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_train_loss = 0
            epoch_train_loss_1 = 0
            epoch_train_acc = 0
            epoch_train_loss_2 = 0
            epoch_train_rmse = 0


            epoch_val_loss = 0
            epoch_val_loss_1 = 0
            epoch_val_acc = 0
            epoch_val_loss_2 = 0
            epoch_val_rmse = 0

            with tf.variable_scope('Training'):
                for hcqt, f0, zeros, atb in data_generator:

                    step_loss, step_loss_1, step_acc, step_loss_2, step_rmse, summary_str = self.train_model(hcqt, f0, zeros, atb, sess)

                    epoch_train_loss+=step_loss
                    epoch_train_loss_1+=step_loss_1
                    epoch_train_acc+=step_acc
                    epoch_train_loss_2+=step_loss_2
                    epoch_train_rmse+= step_rmse

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_train_loss = epoch_train_loss/batch_num
                epoch_train_loss_1 = epoch_train_loss_1/batch_num
                epoch_train_acc = epoch_train_acc/batch_num
                epoch_train_loss_2 = epoch_train_loss_2/batch_num
                epoch_train_rmse = epoch_train_rmse/batch_num


                print_dict = {"Training Loss 1 ": epoch_train_loss_1}
                print_dict["Training Accuracy"] =  epoch_train_acc
                print_dict["Training Loss 2"] =  epoch_train_loss_2
                print_dict["Training Loss"] =  epoch_train_loss
                print_dict["Training RMSE"] =  epoch_train_rmse

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    # pre_score, acc_score, rec_score = self.validate_model(sess)
                    pre, acc, rec = self.validate_model(sess)
                    # import pdb;pdb.set_trace()
                    print_dict["Validation Precision"] = pre
                    print_dict["Validation Accuracy"] = acc
                    print_dict["Validation Recall"] = rec

                    

                    # for cqt, f0, zeros, atb in val_generator:
                    #     step_loss_1, step_acc, step_loss_2, step_rmse, summary_str = self.validate_model(cqt, f0, zeros, atb, sess)
                    #     epoch_val_loss_1+=step_loss_1
                    #     epoch_val_acc+=step_acc
                    #     epoch_val_loss_2+=step_loss_2
                    #     epoch_val_rmse+= step_rmse

                    #     self.val_summary_writer.add_summary(summary_str, epoch)
                    #     self.val_summary_writer.flush()

                    #     batch_num+=1

                    #     utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    # epoch_val_loss_1 = epoch_val_loss_1/batch_num
                    # epoch_val_acc = epoch_val_acc/batch_num
                    # epoch_val_loss_2 = epoch_val_loss_2/batch_num
                    # epoch_val_rmse = epoch_val_rmse/batch_num


                    # print_dict["Validation Loss 1"] =  epoch_train_loss_1
                    # print_dict["Validation Accuracy"] =  epoch_train_acc
                    # print_dict["Validation Loss 2"] =  epoch_train_loss_2
                    # print_dict["Validation RMSE"] =  epoch_train_rmse


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
                # import pdb;pdb.set_trace()
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self,hcqt, f0, zeros,atb, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: hcqt, self.output_placeholder_1: zeros, self.output_placeholder_2: f0, self.output_placeholder: atb, self.is_train: True}
        
        _,_,_,step_loss,  step_loss_1, step_acc, step_loss_2, step_rmse = sess.run(
            [self.train_function,self.train_function_1,self.train_function_2, self.loss, self.loss_1, self.accuracy, self.nll, self.rmse], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        # import pdb;pdb.set_trace()

        return step_loss, step_loss_1, step_acc, step_loss_2, step_rmse, summary_str

    def validate_model(self, sess):
        """
        Function to train the model for each epoch
        """
        # feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: False}
        #
        # step_loss= sess.run(self.loss, feed_dict=feed_dict)
        # summary_str = sess.run(self.summary, feed_dict=feed_dict)
        # return step_loss, summary_str
        val_list = config.val_list
        start_index = randint(0,len(val_list)-(config.batches_per_epoch_val+1))
        pre_scores = []
        acc_scores = []
        rec_scores = []
        count = 0
        for file_name in val_list[start_index:start_index+config.batches_per_epoch_val]:
            pre, acc, rec = self.validate_file(file_name, sess)
            pre_scores.append(pre)
            acc_scores.append(acc)
            rec_scores.append(rec)
            count+=1
            utils.progress(count, config.batches_per_epoch_val, suffix='validation done')
        pre_score = np.array(pre_scores).mean()
        acc_score = np.array(acc_scores).mean()
        rec_score = np.array(rec_scores).mean()
        return pre_score, acc_score, rec_score

    def validate_model_1(self,cqt, f0, zeros, atb,sess):
        """
        Function to validate the model for each epoch
        """
        feed_dict = {self.input_placeholder: cqt, self.output_placeholder_1: zeros, self.output_placeholder_2: f0, self.is_train: True}
        step_loss_1, step_acc, step_loss_2, step_rmse = sess.run(
            [self.loss, self.accuracy, self.nll, self.rmse], feed_dict=feed_dict)
        # import pdb;pdb.set_trace()

        # booboo, baba = sess.run ([self.output_mean, self.output_std ], feed_dict=feed_dict )
        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return step_loss_1, step_acc, step_loss_2, step_rmse, summary_str
        # val_list = config.val_list
        # # start_index = randint(0,len(val_list)-(config.batches_per_epoch_val+1))
        # pre_scores = []
        # acc_scores = []
        # rec_scores = []
        # count = 0
        # for file_name in val_list:
        #     # [start_index:start_index+config.batches_per_epoch_val]
        #     pre, acc, rec = self.validate_file(file_name, sess)
        #     pre_scores.append(pre)
        #     acc_scores.append(acc)
        #     rec_scores.append(rec)
        #     count+=1
        #     utils.progress(count, config.batches_per_epoch_val, suffix='validation done')
        # pre_score = np.array(pre_scores).mean()
        # acc_score = np.array(acc_scores).mean()
        # rec_score = np.array(rec_scores).mean()
        # return pre_score, acc_score, rec_score

    def eval_all(self, file_name_csv):
        sess = tf.Session()
        self.load_model(sess, config.log_dir)
        val_list = config.val_list
        count = 0
        scores = {}
        for file_name in val_list:
            file_score = self.test_file_all(file_name, sess)
            if count == 0:
                for key, value in file_score.items():
                    scores[key] = [value]
                scores['file_name'] = [file_name]
            else:
                for key, value in file_score.items():
                    scores[key].append(value)
                scores['file_name'].append(file_name)

                # import pdb;pdb.set_trace()
            count += 1
            utils.progress(count, len(val_list), suffix='validation done')
        utils.save_scores_mir_eval(scores, file_name_csv)

        return scores


    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.
        """

        with tf.variable_scope('Model') as scope:
            self.output_logits = DeepSalience(self.input_placeholder, self.is_train)
            self.outputs = tf.nn.sigmoid(self.output_logits)

        with tf.variable_scope('Model_1') as scope:
            self.output_logits_1 = DeepSalience_1(self.output_placeholder, self.is_train)
            self.outputs_1 = tf.nn.sigmoid(self.output_logits_1)
        with tf.variable_scope('Model_2') as scope:
            self.output_mean, self.output_std = DeepSalience_2(self.output_placeholder, self.is_train)
            # self.output_mean = self.output_mean * self.outputs_1
            # self.output_std = self.output_std * self.outputs_1



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = Voc_Sep()
    model.extract_file('locus_0024.hdf5', 3)

if __name__ == '__main__':
    test()
