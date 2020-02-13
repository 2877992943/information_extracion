# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
#import bert_utils_albert as bert_utils
import tensorflow

from tensorflow.keras import layers
import tensorflow.compat.v1 as tf1




class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape): # 1st time call will automatically build -> get w variables
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


# class Norm(layers.Layer):
#     def __init__(self):
#         super(Norm,self).__init__()
#         self.layer=tf.keras.layers.LayerNormalization(name='LayerNorm')
#     def call(self,inputs):
#         return self.layer(inputs)


# def Norm():
#     return tf1.layers.BatchNormalization(name='LayerNorm')



class Norm_and_dropout(layers.Layer):
    def __init__(self):
        super(Norm_and_dropout,self).__init__()
        self.norm_l=Norm()
    def call(self,inputs,dropout_prob):
        output_tensor=self.norm_l(inputs)
        output_tensor = dropout(output_tensor, dropout_prob)
        return output_tensor



# class Norm_and_dropout(object):
#     def __init__(self):
#         super(Norm_and_dropout,self).__init__()
#         self.norm_l=Norm()
#     def call(self,inputs,dropout_prob):
#         output_tensor=self.norm_l(inputs)
#         output_tensor = dropout(output_tensor, dropout_prob)
#         tmp = tensorflow.compat.v1.trainable_variables()
#         scop=tensorflow.compat.v1.get_default_graph().get_name_scope()
#         return output_tensor




class Dense(layers.Layer):
    def __init__(self,name=None,
                kernel_initializer=None,
                units=None,
                activation=None,
                 input_dim=None):
        super(Dense,self).__init__()
        self.units=units
        if input_dim==None:
            self.layer=tf.keras.layers.Dense(name=name,
                                         kernel_initializer=kernel_initializer,
                                         units=units,
                                         activation=activation)
        else:
            self.layer = tf.keras.layers.Dense(name=name,
                                               kernel_initializer=kernel_initializer,
                                               units=units,
                                               activation=activation,
                                               input_shape=(input_dim,))

    def call(self,inputs):
        #print (inputs.shape,self.units)
        ret= self.layer(inputs)
        return ret


# def Dense(name=None,
#                 kernel_initializer=None,
#                 units=None,
#                 activation=None,
#                   ):
#
#
#     return tf1.layers.Dense(units=units,
#                             activation=activation,
#                            kernel_initializer=kernel_initializer,
#                            name=name)

class Norm(layers.Layer):
    def __init__(self):
        super(Norm,self).__init__()
        self.l=tf.keras.layers.LayerNormalization(name='LayerNorm')
    def call(self,inputs):
        return self.l(inputs)

####



def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1 - (1.0 - dropout_prob))
  return output


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(input=tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape




def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor,name='')

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.compat.v1.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))