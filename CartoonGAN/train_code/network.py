import tensorflow as tf
from tensorflow.python.ops import variable_scope, image_ops_impl
from tensorflow.keras import layers
import tf_slim as slim

from white_box.train_code import layers


def resblock(inputs, out_channel=32, name='resblock'):
    with variable_scope.variable_scope(name):
        x = slim.convolution2d(inputs, out_channel, [3, 3], activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3], activation_fn=None, scope='conv2')

        return x + inputs


def generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with variable_scope.variable_scope(name, reuse=reuse):
        x = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, channel * 2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 2, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, channel * 4, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 4, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        for idx in range(num_blocks):
            x = resblock(x, out_channel=channel * 4, name='block_{}'.format(idx))

        x = slim.conv2d_transpose(x, channel * 2, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel * 2, [3, 3], activation_fn=None)

        x = tf.nn.leaky_relu(x)

        x = slim.conv2d_transpose(x, channel, [3, 3], stride=2, activation_fn=None)
        x = slim.convolution2d(x, channel, [3, 3], activation_fn=None)
        x = tf.nn.leaky_relu(x)

        x = slim.convolution2d(x, 3, [7, 7], activation_fn=None)

        return x


def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with variable_scope.variable_scope(name, reuse=reuse):
        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = image_ops_impl.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = image_ops_impl.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        return x4


def disc_bn(x, scale=1, channel=32, is_training=True, name='discriminator', patch=True, reuse=False):
    with variable_scope.variable_scope(name, reuse=reuse):
        for idx in range(3):
            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], stride=2, activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

        if patch:
            x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def disc_sn(x, scale=1, channel=32, patch=True, name='discriminator', reuse=False):
    with variable_scope.variable_scope(name):
        for idx in range(3):
            x = layers.conv_spectral_norm(x, channel * 2 ** idx, [3, 3], stride=2, name='conv{}_1'.format(idx))
            x = tf.nn.leaky_relu(x)
            x = layers.conv_spectral_norm(x, channel * 2 ** idx, [3, 3], name='conv{}_2'.format(idx))
            x = tf.nn.leaky_relu(x)

        if patch:
            x = layers.conv_spectral_norm(x, 1, [1, 1], name='conv_out'.format(idx))
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def disc_ln(x, channel=32, is_training=True, name='discriminator', patch=True, reuse=False):
    with variable_scope.variable_scope(name, reuse=reuse):
        for idx in range(3):
            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], stride=2, activation_fn=None)
            x = layers.LayerNormalization(x)
            x = tf.nn.leaky_relu(x)

            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], activation_fn=None)
            x = layers.LayerNormalization(x)
            x = tf.nn.leaky_relu(x)

        if patch:
            x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


if __name__ == '__main__':
    pass
