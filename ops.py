"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.contrib import layers
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor

from utils import *

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, is_training, scope):
    return slim.batch_norm( x,
                            decay=0.9,
                            updates_collections=None,
                            epsilon=1e-5,
                            scale=True,
                            is_training=is_training,
                            scope=scope)



def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def resize_conv(input, out_h, out_w, out_dim, method=1, name='rc'):
    with tf.variable_scope(name):
        resize = tf.image.resize_images(input, [out_h, out_w], method=method)  # Resize NearestNeighbor Method
        padding = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.int32)
        padding = np.reshape(padding, [4, 2])
        pad = tf.pad(resize, paddings=padding)
        print('padding', pad.shape)
        conv = conv2d(resize, out_dim, 3, 3, 1, 1)
        print('resize_conv', conv)
    return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def fc( inputs,
        num_outputs,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)


def encoder(input_tensor, output_size):
    '''Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
    '''
    net = tf.reshape(input_tensor, [-1, 64, 64, 1])
    net = layers.conv2d(net, 32, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 64, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    print(net)
    return layers.fully_connected(net, output_size, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer())

def last_conv(input_, stddev=0.02, reuse=False, name=None):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = tf.get_variable('weights', [4, 4, input_.get_shape()[3], 1],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    return output

def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = tf.get_variable('weights1', [3, 3, input.get_shape()[3], k],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))

      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _instance_norm(conv1)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _instance_norm(conv2)
    output = input+normalized2
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset


def uk(input, k, reuse=False, name=None, output_size=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
      with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[3, 3, k, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _instance_norm(fsconv)
    output = tf.nn.relu(normalized)
    return output

def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))