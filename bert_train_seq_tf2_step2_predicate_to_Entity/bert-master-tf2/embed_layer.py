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



class embedding_lookup_init(tf.Module):
    def __init__(self,vocab_size,
                    #input_ids,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     #use_one_hot_embeddings=False
                    ):
        super(embedding_lookup_init).__init__()

        self.embedding_table = tf.compat.v1.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))

    def __call__(self,
             input_ids,
             vocab_size,
             embedding_size=128,
             # initializer_range=0.02,
             # word_embedding_name="word_embeddings",
             use_one_hot_embeddings=False,
             #inputs=''
             ):
        # This function assumes that the input is of shape [batch_size, seq_length,
        # num_inputs].
        #
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])



        flat_input_ids = tf.reshape(input_ids, [-1])
        if use_one_hot_embeddings:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)

        input_shape = get_shape_list(input_ids, name='')

        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * embedding_size])
        # return (output, embedding_table)
        return output  # [2 3 512]




class embedding_postprocessor_init(tf.Module):

    def __init__(self,
                 input_shape,
                use_token_type=False,
                #token_type_ids=None,
                token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            #dropout_prob=0.1
                            ):
        super(embedding_postprocessor_init).__init__()

        self.token_type_table, self.full_position_embeddings = None, None
        # input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # output = input_tensor

        if use_token_type:
            # if token_type_ids is None:
            #     raise ValueError("`token_type_ids` must be specified if"
            #                  "`use_token_type` is True.")
            self.token_type_table = tf.compat.v1.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, width],
                initializer=create_initializer(initializer_range))


        if use_position_embeddings:
            assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                self.full_position_embeddings = tf.compat.v1.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, width],
                    initializer=create_initializer(initializer_range))

        #####
        self.embedding_norm_drop = Norm_and_dropout()


    def __call__(self,input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            #token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            #position_embedding_name="position_embeddings",
                            #initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            #inputs=''
                            ):
       input_shape = get_shape_list(input_tensor, expected_rank=3, name='')
       batch_size = input_shape[0]
       seq_length = input_shape[1]
       width = input_shape[2]

       output = input_tensor

       if use_token_type:
           if token_type_ids is None:
               raise ValueError("`token_type_ids` must be specified if"
                                "`use_token_type` is True.")

           flat_token_type_ids = tf.reshape(token_type_ids, [-1])
           one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
           token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
           token_type_embeddings = tf.reshape(token_type_embeddings,
                                              [batch_size, seq_length, width])
           output += token_type_embeddings

       if use_position_embeddings:
           assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)



           position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                          [seq_length, -1])
           num_dims = len(output.shape.as_list())



           position_broadcast_shape = []
           for _ in range(num_dims - 2):
               position_broadcast_shape.append(1)
           position_broadcast_shape.extend([seq_length, width])
           position_embeddings = tf.reshape(position_embeddings,
                                            position_broadcast_shape)
           output += position_embeddings

           #output = self.layer_norm_and_dropout(output, dropout_prob)
           output=self.embedding_norm_drop(output,dropout_prob)
       return output





