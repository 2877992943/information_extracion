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

from tensorflow.keras import layers
from some_fn_keras import *



def attention_layer(query_layer_l,
                    key_layer_l,
                    value_layer_l,
                    from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3],name='')
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3],name='')

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  # query_layer = tf.compat.v1.layers.dense(
  #     from_tensor_2d,
  #     num_attention_heads * size_per_head,
  #     activation=query_act,
  #     name="query",
  #     kernel_initializer=create_initializer(initializer_range))

  # query_layer_l=Dense(name='query',
  #                     kernel_initializer=create_initializer(initializer_range),
  #                     units=num_attention_heads * size_per_head,
  #                     activation=query_act)

  query_layer=query_layer_l(from_tensor_2d)

  # `key_layer` = [B*T, N*H]
  # key_layer = tf.compat.v1.layers.dense(
  #     to_tensor_2d,
  #     num_attention_heads * size_per_head,
  #     activation=key_act,
  #     name="key",
  #     kernel_initializer=create_initializer(initializer_range))
  # key_layer_l = Dense(name='key',
  #                       kernel_initializer=create_initializer(initializer_range),
  #                       units=num_attention_heads * size_per_head,
  #                       activation=key_act)

  key_layer = key_layer_l(to_tensor_2d)


  # `value_layer` = [B*T, N*H]
  # value_layer = tf.compat.v1.layers.dense(
  #     to_tensor_2d,
  #     num_attention_heads * size_per_head,
  #     activation=value_act,
  #     name="value",
  #     kernel_initializer=create_initializer(initializer_range))
  # value_layer_l = Dense(name='value',
  #                     kernel_initializer=create_initializer(initializer_range),
  #                     units=num_attention_heads * size_per_head,
  #                     activation=value_act)

  value_layer = value_layer_l(to_tensor_2d)


  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer



#class each_layer(layers.Layer):
class each_layer(tf.Module):
    def __init__(self,
                 input_shape,
                 # attention_mask=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 do_return_all_layers=False
                 ):
        super(each_layer).__init__()
        # if hidden_size % num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        with tf.compat.v1.variable_scope("attention"):
            attention_heads = []
            with tf.compat.v1.variable_scope("self"):
                #### move out
                self.query_layer_l = Dense(name='query',
                                      kernel_initializer=create_initializer(initializer_range),
                                      units=num_attention_heads * attention_head_size,
                                      )
                self.key_layer_l = Dense(name='key',
                                    kernel_initializer=create_initializer(initializer_range),
                                    units=num_attention_heads * attention_head_size,
                                    )
                self.value_layer_l = Dense(name='value',
                                      kernel_initializer=create_initializer(initializer_range),
                                      units=num_attention_heads * attention_head_size,
                                      )


            with tf.compat.v1.variable_scope("output"):
                self.dense_l_out = Dense(kernel_initializer=create_initializer(initializer_range),
                                    units=hidden_size)

                self.norm_l = Norm()


        # The activation is only applied to the "intermediate" hidden layer.
        with tf.compat.v1.variable_scope("intermediate"):
            self.dense_l_intermediate = Dense(kernel_initializer=create_initializer(initializer_range),
                                         units=intermediate_size,
                                         activation=intermediate_act_fn)


            # Down-project back to `hidden_size` then add the residual.
        with tf.compat.v1.variable_scope("output"):
            self.dense_l_out2 = Dense(kernel_initializer=create_initializer(initializer_range),
                                 units=hidden_size)

            self.norm_l_layer_output = Norm()






#class get_tranformer_layer(layers.Layer):
class get_tranformer_layer(tf.Module):
    def __init__(self,
                    input_shape,
                      #attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
        super(get_tranformer_layer).__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
          #input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, hidden_size))

          # We keep the representation as a 2D tensor to avoid re-shaping it back and
          # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
          # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
          # help the optimizer.
          #prev_output = reshape_to_matrix(input_tensor)


        self.layer0=each_layer(input_shape,
                 # attention_mask=None,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 intermediate_act_fn,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 initializer_range,
                 do_return_all_layers)
        self.layer1 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer2 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer3= each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer4 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer5= each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer6 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer7 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer8 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer9 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer10 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)
        self.layer11 = each_layer(input_shape,
                                 # attention_mask=None,
                                 hidden_size,
                                 num_hidden_layers,
                                 num_attention_heads,
                                 intermediate_size,
                                 intermediate_act_fn,
                                 hidden_dropout_prob,
                                 attention_probs_dropout_prob,
                                 initializer_range,
                                 do_return_all_layers)

    def __call__(self,
              input_tensor,
              attention_mask=None,
              hidden_size=768,
              num_hidden_layers=12,
              num_attention_heads=12,
              intermediate_size=3072,
              intermediate_act_fn=gelu,
              hidden_dropout_prob=0.1,
              attention_probs_dropout_prob=0.1,
              initializer_range=0.02,
              do_return_all_layers=False,
              #inputs=''
             ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        input_shape = get_shape_list(input_tensor, expected_rank=3, name='')
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, hidden_size))

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(input_tensor)  # [ 2 3 512]-> [6 512]


        all_layer_outputs=[]

        # #### 1 layer


        def each_l_fn(prev_output,self_layer_n):
            layer_input = prev_output


            with tf.compat.v1.variable_scope("attention"):
                attention_heads = []
                with tf.compat.v1.variable_scope("self"):
                    #### move out
                    attention_head = attention_layer(
                        self_layer_n.query_layer_l,
                        self_layer_n.key_layer_l,
                        self_layer_n.value_layer_l,
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.compat.v1.variable_scope("output"):
                    # dense_l_out=Dense(kernel_initializer=create_initializer(initializer_range),
                    #                   units=hidden_size)

                    attention_output = self_layer_n.dense_l_out(attention_output)
                    # attention_output = tf.compat.v1.layers.dense(
                    #     attention_output,
                    #     hidden_size,
                    #     kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)

                    # attention_output = Norm(attention_output + layer_input)
                    # norm_l=Norm()

                    attention_output = self_layer_n.norm_l(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.compat.v1.variable_scope("intermediate"):
                # intermediate_output = tf.compat.v1.layers.dense(
                #     attention_output,
                #     intermediate_size,
                #     activation=intermediate_act_fn,
                #     kernel_initializer=create_initializer(initializer_range))
                # dense_l_intermediate=Dense(kernel_initializer=create_initializer(initializer_range),
                #                            units=intermediate_size,
                #                            activation=intermediate_act_fn)


                intermediate_output = self_layer_n.dense_l_intermediate(attention_output)

                # Down-project back to `hidden_size` then add the residual.
            with tf.compat.v1.variable_scope("output"):
                # layer_output = tf.compat.v1.layers.dense(
                #     intermediate_output,
                #     hidden_size,
                #     kernel_initializer=create_initializer(initializer_range))
                # dense_l_out2=Dense(kernel_initializer=create_initializer(initializer_range),
                #                    units=hidden_size)

                layer_output = self_layer_n.dense_l_out2(intermediate_output)

                layer_output = dropout(layer_output, hidden_dropout_prob)
                # norm_l_layer_output = Norm() ##

                layer_output = self_layer_n.norm_l_layer_output(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
                return prev_output



        #####

        # #### 1 layer

        prev_output=each_l_fn(prev_output,self.layer0)
        prev_output = each_l_fn(prev_output, self.layer1)
        prev_output = each_l_fn(prev_output, self.layer2)
        prev_output = each_l_fn(prev_output, self.layer3)
        prev_output = each_l_fn(prev_output, self.layer4)
        prev_output = each_l_fn(prev_output, self.layer5)
        prev_output = each_l_fn(prev_output, self.layer6)
        prev_output = each_l_fn(prev_output, self.layer7)
        prev_output = each_l_fn(prev_output, self.layer8)
        prev_output = each_l_fn(prev_output, self.layer9)
        prev_output = each_l_fn(prev_output, self.layer10)
        prev_output = each_l_fn(prev_output, self.layer11)
        ###


        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output






