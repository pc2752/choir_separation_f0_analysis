import tensorflow as tf
from modules_tf import DeepSalience
import config
from data_pipeline import data_gen
import time, os
import utils
import h5py
import numpy as np
import mir_eval
from random import randint

class DeepSal(object):
    def __init__(self):
        """
        Function to initialize the synthesizer class.
        Will see what parameters need to be initialized here.
        Should initialize:
        1) modes and epochs, load from config. 
        2) Placeholders
        3) Generator and discriminator/critic models.

        """
        # with tf.Graph().as_default():
        self.get_placeholders()
        self.model()




    def read_input_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        feat_file = h5py.File(config.feats_dir + file_name)
        atb = feat_file['atb'][()]

        atb = atb[:, 1:]

        hcqt = feat_file['voc_hcqt'][()]

        feat_file.close()

        in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*360))
        in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
                                                      6, 360)
        in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)
        # in_batches_atb, nchunks_in = utils.generate_overlapadd(atb)

        return in_batches_hcqt, atb, nchunks_in


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
    def test_file(self, file_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess)
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

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 360, 6),name='input_placeholder')
        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 360),name='output_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def load_model(self, sess):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)




        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output_logits))

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_function = self.optimizer.minimize(self.loss, global_step = self.global_step)


    def get_summary(self, sess):
        """
        Gets the summaries and summary writers for the losses.
        """
        self.loss_summary = tf.summary.scalar('final_loss', self.loss)
        self.train_summary_writer = tf.summary.FileWriter(config.log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(config.log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()
    def save_model(self, sess, epoch):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))
        # import pdb;pdb.set_trace()

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess)
        self.get_summary(sess)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = data_gen()
            start_time = time.time()


            batch_num = 0
            epoch_train_loss = 0


            with tf.variable_scope('Training'):
                for ins, outs in data_generator:

                    step_loss, summary_str = self.train_model(ins, outs, sess)
                    epoch_train_loss+=step_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_train_loss = epoch_train_loss/batch_num
                print_dict = {"Training Loss": epoch_train_loss}

            if (epoch + 1) % config.validate_every == 0:
                pre, acc, rec = self.validate_model(sess)
                print_dict["Validation Precision"] = pre
                print_dict["Validation Accuracy"] = acc
                print_dict["Validation Recall"] = rec

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1)

    def train_model(self, ins, outs, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: True}
        _, step_loss= sess.run(
            [self.train_function, self.loss], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return step_loss, summary_str

    def validate_model(self, sess):
        """
        Function to train the model for each epoch
        """
        # feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: False}
        #
        # step_loss= sess.run(self.loss, feed_dict=feed_dict)
        # summary_str = sess.run(self.summary, feed_dict=feed_dict)
        # return step_loss, summary_str
        val_list = [x for x in os.listdir(config.feats_dir) if x.endswith('.hdf5') and x.__contains__('1')]
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




    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('Model') as scope:
            self.output_logits = DeepSalience(self.input_placeholder, self.is_train)
            self.outputs = tf.nn.sigmoid(self.output_logits)




def test():
    model = DeepSal()
    model.extract_f0_file('locus_0101.hdf5')

if __name__ == '__main__':
    test()




