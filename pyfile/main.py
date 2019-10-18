# -*- coding: utf-8 -*-
import tensorflow as tf
from cgan import CGAN
if __name__ == '__main__':

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True

    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=64, sample_interval=200)