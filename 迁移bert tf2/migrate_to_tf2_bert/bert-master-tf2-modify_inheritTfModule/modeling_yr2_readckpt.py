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
from some_fn_keras import Norm_and_dropout,Norm,Dense

from embed_layer import *

from transformer_layer import *


from read_ckpt_utils_yr import *

class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


#class BertModel(layers.Layer):
class BertModel(tf.Module):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               #input_ids, # move
               batch_size,
               seq_length,
               #input_mask=None, # move
               #token_type_ids=None, #move
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """

    super(BertModel).__init__()
    config = copy.deepcopy(config)
    self.config=config
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    # input_shape = get_shape_list(input_ids, expected_rank=2)
    # batch_size = input_shape[0]
    # seq_length = input_shape[1]


    # #### move to call
    # if input_mask is None:
    #   input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    #
    # if token_type_ids is None:
    #   token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.compat.v1.variable_scope(scope, default_name="bert"):
      with tf.compat.v1.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        self.embedding_lookup_l=embedding_lookup_init(
            #input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            #use_one_hot_embeddings=use_one_hot_embeddings
                )
        # ### move to call
        # self.embedding_output=self.embedding_lookup_l.call1(
        #     input_ids=input_ids,
        #     vocab_size=config.vocab_size,
        #     embedding_size=config.hidden_size,
        #     use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_postprocessor_l=embedding_postprocessor_init(
            #input_tensor=self.embedding_output,
            input_shape=[batch_size,seq_length,config.hidden_size],
            use_token_type=True,
            #token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            #dropout_prob=config.hidden_dropout_prob
        )

        ### move to call
        # self.embedding_output=self.embedding_postprocessor_l.call1(
        #                             input_tensor=self.embedding_output,
        #                              use_token_type=True,
        #                              token_type_ids=token_type_ids,
        #                              token_type_vocab_size=config.type_vocab_size,
        #                              # token_type_embedding_name="token_type_embeddings",
        #                              use_position_embeddings=True,
        #                              # position_embedding_name="position_embeddings",
        #                              # initializer_range=0.02,
        #                              max_position_embeddings=config.max_position_embeddings,
        #                              dropout_prob=config.hidden_dropout_prob
        #                              )


        # emb output [2 3 512]
      with tf.compat.v1.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.


        # move to call
        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask) # [2 3seq 3seq]

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.get_tranformer_layer=get_tranformer_layer(input_shape=[batch_size,seq_length,config.hidden_size],
            #attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      #   self.all_encoder_layers = self.get_tranformer_layer.call1(
      #
      #       input_tensor=self.embedding_output,
      #       attention_mask=attention_mask,
      #       hidden_size=config.hidden_size,
      #       num_hidden_layers=config.num_hidden_layers,
      #       num_attention_heads=config.num_attention_heads,
      #       intermediate_size=config.intermediate_size,
      #       intermediate_act_fn=get_activation(config.hidden_act),
      #       hidden_dropout_prob=config.hidden_dropout_prob,
      #       attention_probs_dropout_prob=config.attention_probs_dropout_prob,
      #       initializer_range=config.initializer_range,
      #       do_return_all_layers=True)
      #
      # self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.



      # with tf.compat.v1.variable_scope("pooler"):
      #   # We "pool" the model by simply taking the hidden state corresponding
      #   # to the first token. We assume that this has been pre-trained
      #   first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
      #   self.pooled_output = tf.compat.v1.layers.dense(
      #       first_token_tensor,
      #       config.hidden_size,
      #       activation=tf.tanh,
      #       kernel_initializer=create_initializer(config.initializer_range))



  def __call__(self,
            batch_size,
            seq_length,
            input_ids, # move
            input_mask=None, # move
            token_type_ids=None, #move
            use_one_hot_embeddings=False
            ):
      if input_mask is None:
        input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

      if token_type_ids is None:
        token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
      embedding_output = self.embedding_lookup_l(
          input_ids=input_ids,
          vocab_size=self.config.vocab_size,
          embedding_size=self.config.hidden_size,
          use_one_hot_embeddings=use_one_hot_embeddings,
      )
      #
      embedding_output = self.embedding_postprocessor_l(
          input_tensor=embedding_output,
          use_token_type=True,
          token_type_ids=token_type_ids,
          token_type_vocab_size=self.config.type_vocab_size,
          # token_type_embedding_name="token_type_embeddings",
          use_position_embeddings=True,
          # position_embedding_name="position_embeddings",
          # initializer_range=0.02,
          max_position_embeddings=self.config.max_position_embeddings,
          dropout_prob=self.config.hidden_dropout_prob,

      )

      attention_mask = create_attention_mask_from_input_mask(
          input_ids, input_mask)  # [2 3seq 3seq]

      all_encoder_layers = self.get_tranformer_layer(
          input_tensor=embedding_output,
          attention_mask=attention_mask,
          hidden_size=self.config.hidden_size,
          num_hidden_layers=self.config.num_hidden_layers,
          num_attention_heads=self.config.num_attention_heads,
          intermediate_size=self.config.intermediate_size,
          intermediate_act_fn=get_activation(self.config.hidden_act),
          hidden_dropout_prob=self.config.hidden_dropout_prob,
          attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
          initializer_range=self.config.initializer_range,
          do_return_all_layers=True)

      sequence_output = all_encoder_layers[-1]
      return sequence_output




  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


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


# def layer_norm(input_tensor, name=None):
#   """Run layer normalization on the last dimension of the tensor."""
#   return tf.contrib.layers.layer_norm(
#       inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
#
#
# def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
#   """Runs layer normalization followed by dropout."""
#   output_tensor = layer_norm(input_tensor, name)
#   output_tensor = dropout(output_tensor, dropout_prob)
#   return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)



def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3],name='')
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2,name='')
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


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





def layer_update_weight(submodule,layer_i):
    subm11 = submodule.query_layer_l
    namell = ['bert/encoder/layer_%d/attention/self/query/kernel'%layer_i,
              'bert/encoder/layer_%d/attention/self/query/bias'%layer_i]
    from_submodel_set_weights(subm11, namell, varDict)
    #
    m12 = submodule.key_layer_l
    namell = ['bert/encoder/layer_%d/attention/self/key/kernel'%layer_i,
              'bert/encoder/layer_%d/attention/self/key/bias'%layer_i]
    from_submodel_set_weights(m12, namell, varDict)
    #
    m13 = submodule.value_layer_l
    namell = ['bert/encoder/layer_%d/attention/self/value/kernel'%layer_i,
              'bert/encoder/layer_%d/attention/self/value/bias'%layer_i]
    from_submodel_set_weights(m13, namell, varDict)
    #
    m14 = submodule.dense_l_out
    namell = ['bert/encoder/layer_%d/attention/output/dense/kernel'%layer_i,
              'bert/encoder/layer_%d/attention/output/dense/bias'%layer_i]
    from_submodel_set_weights(m14, namell, varDict)

    m15 = submodule.norm_l
    namell = ['bert/encoder/layer_%d/attention/output/LayerNorm/gamma'%layer_i,
              'bert/encoder/layer_%d/attention/output/LayerNorm/beta'%layer_i]
    from_submodel_set_weights(m15, namell, varDict)

    m16 = submodule.dense_l_intermediate
    namell = ['bert/encoder/layer_%d/intermediate/dense/kernel'%layer_i,
              'bert/encoder/layer_%d/intermediate/dense/bias'%layer_i]
    from_submodel_set_weights(m16, namell, varDict)

    m17 = submodule.dense_l_out2
    namell = ['bert/encoder/layer_%d/output/dense/kernel'%layer_i,
              'bert/encoder/layer_%d/output/dense/bias'%layer_i]
    from_submodel_set_weights(m17, namell, varDict)

    m18 = submodule.norm_l_layer_output
    namell = ['bert/encoder/layer_%d/output/LayerNorm/gamma'%layer_i,
              'bert/encoder/layer_%d/output/LayerNorm/beta'%layer_i]
    from_submodel_set_weights(m18, namell, varDict)
    return submodule



########
if __name__=='__main__':
    f_init_ckpt='/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'
    varDict, ll = read_var_from_ckpt(f_init_ckpt)
    l1=[n for n in varDict.keys() if 'bert/encoder/layer_0' in n]
    print ('\n'.join(l1))








    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    import modeling
    f='/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_config.json'
    bert_config = modeling.BertConfig.from_json_file(f)
    bert_config.type_vocab_size =26   # 修改


    bertEncode=BertModel(bert_config,
               True,
               #input_ids, # move
               batch_size=2,
               seq_length=3,
               #input_mask=input_mask, # move
               #token_type_ids=token_type_ids, #move
               use_one_hot_embeddings=False,
               scope=None)


    out=bertEncode(
                    batch_size=2,
                   seq_length=3,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    )


    #########
    # #embed
    m1=bertEncode.embedding_lookup_l.embedding_table #[21128 768] # vars
    var_assign(m1,varDict['bert/embeddings/word_embeddings'])
    #m2=bertEncode.embedding_postprocessor_l.token_type_table #[2,768]# 不需要这个参数
    #m3=bertEncode.embedding_postprocessor_l.full_position_embeddings #[512 768]
    m4_std_mean=bertEncode.embedding_postprocessor_l.embedding_norm_drop # submodule
    #m4_v=from_submodel_get_weights(m4_std_mean)

    namell=['bert/embeddings/LayerNorm/gamma','bert/embeddings/LayerNorm/beta']
    from_submodel_set_weights(m4_std_mean,namell,varDict)
    #m4_v_=from_submodel_get_weights(m4_std_mean) # different from m4_v

    ###########
    # # encode
    ## layer 0
    subm11=bertEncode.get_tranformer_layer.layer0.query_layer_l
    namell = ['bert/encoder/layer_0/attention/self/query/kernel',
              'bert/encoder/layer_0/attention/self/query/bias']
    from_submodel_set_weights(subm11, namell, varDict)
    #
    m12=bertEncode.get_tranformer_layer.layer0.key_layer_l
    namell = ['bert/encoder/layer_0/attention/self/key/kernel',
              'bert/encoder/layer_0/attention/self/key/bias']
    from_submodel_set_weights(m12, namell, varDict)
    #
    m13=bertEncode.get_tranformer_layer.layer0.value_layer_l
    namell = ['bert/encoder/layer_0/attention/self/value/kernel',
              'bert/encoder/layer_0/attention/self/value/bias']
    from_submodel_set_weights(m13, namell, varDict)
    #
    m14=bertEncode.get_tranformer_layer.layer0.dense_l_out
    namell=['bert/encoder/layer_0/attention/output/dense/kernel',
            'bert/encoder/layer_0/attention/output/dense/bias']
    from_submodel_set_weights(m14, namell, varDict)


    m15=bertEncode.get_tranformer_layer.layer0.norm_l
    namell=['bert/encoder/layer_0/attention/output/LayerNorm/gamma',
            'bert/encoder/layer_0/attention/output/LayerNorm/beta']
    from_submodel_set_weights(m15, namell, varDict)

    m16=bertEncode.get_tranformer_layer.layer0.dense_l_intermediate
    namell=['bert/encoder/layer_0/intermediate/dense/kernel',
            'bert/encoder/layer_0/intermediate/dense/bias']
    from_submodel_set_weights(m16, namell, varDict)

    m17=bertEncode.get_tranformer_layer.layer0.dense_l_out2
    namell=['bert/encoder/layer_0/output/dense/kernel',
            'bert/encoder/layer_0/output/dense/bias']
    from_submodel_set_weights(m17, namell, varDict)


    m18=bertEncode.get_tranformer_layer.layer0.norm_l_layer_output
    namell=['bert/encoder/layer_0/output/LayerNorm/gamma',
            'bert/encoder/layer_0/output/LayerNorm/beta']
    from_submodel_set_weights(m18, namell, varDict)
    ###

    print ('')


    ##### layer1
    l1_subm=bertEncode.get_tranformer_layer.layer1
    l1_subm=layer_update_weight(l1_subm,1)
    #m_after=bertEncode.get_tranformer_layer.layer1.norm_l_layer_output
    #w=from_submodel_get_weights(m_after)
    print ('')
    #### layer2
    l2_subm = bertEncode.get_tranformer_layer.layer2
    l2_subm = layer_update_weight(l2_subm, 2)

    l3_subm = bertEncode.get_tranformer_layer.layer3
    l3_subm = layer_update_weight(l3_subm, 3)

    l4_subm = bertEncode.get_tranformer_layer.layer4
    l4_subm = layer_update_weight(l4_subm, 4)

    l5_subm = bertEncode.get_tranformer_layer.layer5
    l5_subm = layer_update_weight(l5_subm, 5)

    l6_subm = bertEncode.get_tranformer_layer.layer6
    l6_subm = layer_update_weight(l6_subm, 6)

    l7_subm = bertEncode.get_tranformer_layer.layer7
    l7_subm = layer_update_weight(l7_subm, 7)

    l8_subm = bertEncode.get_tranformer_layer.layer8
    l8_subm = layer_update_weight(l8_subm, 8)

    l9_subm = bertEncode.get_tranformer_layer.layer9
    l9_subm = layer_update_weight(l9_subm, 9)

    l10_subm = bertEncode.get_tranformer_layer.layer10
    l10_subm = layer_update_weight(l10_subm, 10)

    l11_subm = bertEncode.get_tranformer_layer.layer11
    l11_subm = layer_update_weight(l11_subm, 11)




    #### save model object-based
    ckpt = tf.train.Checkpoint(net=bertEncode)
    ckpt.save('../model_encode_ckpt/')

