from __future__ import print_function, division

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist


class CGAN():

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_class = 10
        self.latent_dim = 100

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        label = tf.keras.layers.Input(shape=(1,))
        img = self.generator([noise, label])

        valid = self.discriminator([img, label])

        self.combined_model = tf.keras.Model([noise, label], valid)
        self.combined_model.compile(loss=['binary_crossentropy'],
                                    optimizer=optimizer)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, Y_train), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        Y_train = Y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], Y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the Generator
            g_loss = self.combined_model.train_on_batch([noise, sampled_labels], valid)

            print("epoch:%d/%d [D loss: %f, acc.: %.2f] [G loss: %f]" % (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def build_discriminator(self):
        d_model = tf.keras.Sequential()
        d_model.add(tf.keras.layers.Dense(512, input_dim=np.prod(self.img_shape)))
        d_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        d_model.add(tf.keras.layers.Dense(512))
        d_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        d_model.add(tf.keras.layers.Dropout(rate=0.4))
        d_model.add(tf.keras.layers.Dense(512))
        d_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        d_model.add(tf.keras.layers.Dropout(rate=0.4))
        d_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        d_model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape)
        label = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        label_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(self.num_class, np.prod(self.img_shape))(label))
        flat_img = tf.keras.layers.Flatten()(img)

        model_input = tf.keras.layers.multiply([flat_img, label_embedding])

        validity = d_model(model_input)

        return tf.keras.Model([img, label], validity)

    def build_generator(self):
        g_model = tf.keras.Sequential()

        g_model.add(tf.keras.layers.Dense(256, input_dim=self.latent_dim))
        g_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        g_model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        g_model.add(tf.keras.layers.Dense(512))
        g_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        g_model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        g_model.add(tf.keras.layers.Dense(1024))
        g_model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        g_model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        g_model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        g_model.add(tf.keras.layers.Reshape(self.img_shape))
        g_model.summary()

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        label = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        label_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(self.num_class, self.latent_dim)(label))
        model_input = tf.keras.layers.multiply([noise, label_embedding])
        img = g_model(model_input)

        return tf.keras.Model([noise, label], img)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r*c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title('Digit: %d' % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/gen_%d.png" % epoch)
        plt.close()
