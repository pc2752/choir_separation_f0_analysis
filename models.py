import tensorflow as tf
from modules_tf import DeepSalience
import config
from data_pipeline import data_gen
import time, os
import utils

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
        self.get_placeholders()
        self.model()




    def read_input_file(self, file_name, synth_mode):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on maode. 
        Mode 0 is for direct synthesis from frame-wise phoneme and note annotations, like in Nitech.
        Mode 1 is for partial synthesis, from frame-wise phoeneme and f0 annotations, like in NUS.
        Mode 2 is for sem-partial synthesis, from loose framewise phoneme and note annotation, like in Ayesha recordings.
        Mode 3 is from indirect synthesis, etracting features from the audio recording and syntthesizing. 
        """

    def synth_file(self, input_features):
        """
        Function to synthesize a singing voice based on the input features. 
        """
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

    def print_summary(self, epoch_train_loss, epoch_validation_loss, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """
        epoch_train_loss = epoch_train_loss/config.batches_per_epoch_train
        epoch_validation_loss = epoch_validation_loss / config.batches_per_epoch_val

        print('epoch %d: Training Loss = %.10f (%.3f sec)' % (epoch + 1, epoch_train_loss, duration))
        print('        : Validtion Loss = %.10f ' % (epoch_validation_loss))

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

            val_generator = data_gen(mode='val')

            batch_num = 0

            batch_num_val = 0

            epoch_train_loss = 0
            epoch_validation_loss = 0

            with tf.variable_scope('Training'):
                for ins, outs in data_generator:

                    step_loss, summary_str = self.train_model(ins, outs, sess)
                    epoch_train_loss+=step_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1


            with tf.variable_scope('Validation'):
                for ins, outs in val_generator:
                    step_loss, summary_str = self.validate_model(ins, outs, sess)
                    epoch_validation_loss+=step_loss

                    self.val_summary_writer.add_summary(summary_str, epoch)
                    self.val_summary_writer.flush()

                    utils.progress(batch_num_val,config.batches_per_epoch_val, suffix = 'validation done')

                    batch_num_val+=1
                end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(epoch_train_loss, epoch_validation_loss, epoch, end_time-start_time)
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

    def validate_model(self, ins, outs, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: False}

        step_loss= sess.run(self.loss, feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)
        return step_loss, summary_str

    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('Model') as scope:
            self.output_logits = DeepSalience(self.input_placeholder, self.is_train)




def test():
    model = Model()

if __name__ == '__main__':
    test()





