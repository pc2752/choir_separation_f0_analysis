import models
import tensorflow as tf

def main(_):
    model = models.DeepSal()
    model.train()

if __name__ == '__main__':
    tf.app.run(main=main)