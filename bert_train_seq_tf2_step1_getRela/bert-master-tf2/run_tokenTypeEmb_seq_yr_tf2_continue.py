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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling_yr2 as modeling
import optimization
import tokenization
import six

import numpy as np
import tensorflow as tf

import tensorflow.compat.v1 as tf1
from tensorflow import keras

from some_fn_keras import Dense
import json
import sys

eps=0.0000001


# #####使用   decoder


path_to_be_add='../models_src_part_v1v2_v1025/'
import sys
from sys import path
from os.path import abspath, dirname, join
sys.path.insert(0, path_to_be_add)


from official.transformer.model import model_params
from official.transformer.v2 import transformerDecoder_v2
from official.transformer.v2 import metrics



flags = tf1.flags

FLAGS = flags.FLAGS

from some_flags import *






class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s

# class knowledgeExample(object):
#   """A single training/test example for simple sequence classification.
#
#      For examples without an answer, the start and end position are -1.
#   """
#
#   def __init__(self,
#                text,
#                doc_tokens,
#                tokentype,
#                ytoken
#                ):
#       self.text=text
#       self.doc_tokens=doc_tokens
#       self.tokentype=tokentype
#       self.ytoken=ytoken
#
#   def __str__(self):
#     return self.__repr__()
#
#   def __repr__(self):
#     s = ""
#
#     s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
#     if self.text:
#       s += ", start_position: %d" % (self.text)
#     if self.ytoken:
#       s += ", end_position: %d" % (self.ytoken)
#
#     return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               #example_index,
               #doc_span_index,
               #tokens,
               #token_to_orig_map,
               #token_is_max_context,
               input_ids,
               input_mask,
               segment_ids, # token type
               target_ids,
               num_of_target
               #start_position=None,
               #end_position=None,
               #is_impossible=None
               ):
    self.unique_id = unique_id
    #self.example_index = example_index
    #self.doc_span_index = doc_span_index
    #self.tokens = tokens
    #self.token_to_orig_map = token_to_orig_map
    #self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.target_ids=target_ids
    self.num_of_target=num_of_target
    #self.start_position = start_position
    #self.end_position = end_position
    #self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.io.gfile.GFile(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data[:2]:
    for paragraph in entry["paragraphs"][:2]:
      paragraph_text = paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          if FLAGS.version_2_with_negative:
            is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length -
                                               1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
              tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
              continue
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples


def read_knowledge_example(inputfile,is_training):
    examples=[]
    reader=open(inputfile)
    for line in reader.readlines():
        line=line.strip()
        if len(line)==0:continue
        #
        d=json.loads(line)

        spo_list=d['spo_list']
        posseq=d['posseq']
        text=[cell['char'] for cell in posseq];#print (''.join(text))
        postag=[cell['tag'] for cell in posseq];#print (' '.join(postag))
        predicate=[cell['predicate'] for cell in spo_list];#print (' '.join(predicate))
        examples.append({'text':text,'tag':postag,'predicate':predicate})
    return examples



def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn,
                                 tokenizer_y,
                                 max_seq_length_y
                                 ):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      if example_index < 20:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("unique_id: %s" % (unique_id))
        tf.compat.v1.logging.info("example_index: %s" % (example_index))
        tf.compat.v1.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.compat.v1.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          tf.compat.v1.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.compat.v1.logging.info("start_position: %d" % (start_position))
          tf.compat.v1.logging.info("end_position: %d" % (end_position))
          tf.compat.v1.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=example.is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1



def convert_examples_to_features1(examples,
                                  tokenizer,
                                  tokenizer_pos,
                                  max_seq_length,
                                  is_training,
                                 output_fn,
                                 tokenizer_y,
                                 max_seq_length_y
                                 ):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1

  for (example_index, example) in enumerate(examples):

    #query_tokens = tokenizer.tokenize(example.question_text)
    charll,posll,predicatell=example['text'],example['tag'],example['predicate']
    # text_token=tokenizer.tokenize(' '.join(charll))
    # pos_token=tokenizer_pos.tokenize(' '.join(posll))
    # predicate_token=tokenizer_y.tokenize(' '.join(predicatell))
    text_token=[w.lower() for w in charll]
    pos_token=[w.lower() for w in posll]
    predicate_token=[w.lower() for w in predicatell]


    if len(text_token) > max_seq_length-1: # 限制长度
      text_token = text_token[0:max_seq_length]
    if len(pos_token) > max_seq_length-1: # 限制长度
      pos_token = pos_token[0:max_seq_length]
    if len(predicate_token) > max_seq_length_y: # 限制长度
      predicate_token = predicate_token[0:max_seq_length_y]


    text_token=['[CLS]']+text_token
    pos_token=['PAD']+pos_token
    #predicate_token=predicate_token+['EOS']
    predicate_token = predicate_token

    input_ids = tokenizer.convert_tokens_to_ids(text_token)
    segment_ids=tokenizer_pos.convert_tokens_to_ids(pos_token)
    predicate_ids=tokenizer_y.convert_tokens_to_ids(predicate_token)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    # zero pad y seq
    while len(predicate_ids) < max_seq_length_y:
      predicate_ids.append(0)
    assert len(predicate_ids) == max_seq_length_y


    ### num of target
    num_y=len([yi for yi in predicate_ids if yi!=0])

    #tf.logging.info("char: %s" % (' '.join([str(w) for w in input_ids])))
    #tf.logging.info("char mask: %s" % (' '.join([str(w) for w in input_mask])))
    #tf.logging.info("pos: %s" % (' '.join([str(w) for w in segment_ids])))
    #tf.logging.info("predicate: %s" % (' '.join([str(w) for w in predicate_ids])))
    #tf.logging.info("predicate number: %d" % (num_y))




    feature = InputFeatures(
          unique_id=unique_id,
          #example_index=example_index,
          #doc_span_index=doc_span_index,
          #tokens=tokens,
          #token_to_orig_map=token_to_orig_map,
          #token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          target_ids=predicate_ids,
          num_of_target=num_y
          #start_position=start_position,
          #end_position=end_position,
          #is_impossible=example.is_impossible
          )

    # Run callback
    output_fn(feature)

    unique_id += 1
    #if unique_id>10:break  #yr


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings,
                 targets,
                 #GLOBAL_PARAMS_target
                 ):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  #final_hidden = model.get_sequence_output() # [8 384 768]
  first_token=model.get_pooled_output() #[batch step]

  # final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  # batch_size = final_hidden_shape[0]
  # seq_length = final_hidden_shape[1]
  # hidden_size = final_hidden_shape[2]

  ###  decode part
  # my_decoder = transformer_decoder.TransformerDecoder(GLOBAL_PARAMS_target,train=is_training)###??
  # if is_training :
  #   logits = my_decoder(input_ids,final_hidden,targets)
  # else:
  #   logits = my_decoder(input_ids,final_hidden)

  # return logits

  logits_multic=tf.compat.v1.layers.dense(inputs=first_token,name='multi',
                              units=FLAGS.target_vocab_size)

  logits_num = tf.compat.v1.layers.dense(inputs=first_token,name='num_of_class',
                                  units=FLAGS.target_vocab_size)



  return logits_multic,logits_num  # #[batch vocabsz]

  # output_weights = tf.get_variable(
  #     "cls/squad/output_weights", [2, hidden_size],
  #     initializer=tf.truncated_normal_initializer(stddev=0.02))
  #
  # output_bias = tf.get_variable(
  #     "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())
  #
  # final_hidden_matrix = tf.reshape(final_hidden,
  #                                  [batch_size * seq_length, hidden_size])
  # logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  # logits = tf.nn.bias_add(logits, output_bias)
  #
  # logits = tf.reshape(logits, [batch_size, seq_length, 2])
  # logits = tf.transpose(logits, [2, 0, 1])
  #
  # unstacked_logits = tf.unstack(logits, axis=0) # [2, 8batch, 384step]
  #
  # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
  #
  # return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,
                     #GLOBAL_PARAMS_target
                     ):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    target_ids=features['target_ids'] #[batch step]
    num_target=features['num_of_target'] #[batchsz,]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    logits_multic,logits_num = create_model(
        targets=target_ids,
        #GLOBAL_PARAMS_target=GLOBAL_PARAMS_target,
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings) # [batch vocabsz]

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # xentropy, weights = metrics.padded_cross_entropy_loss(
        #     logits, targets, FLAGS.label_smoothing, FLAGS.vocab_size_symptom)
        # loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        ##### loss 1 number of target
        y=tf.one_hot(num_target,depth=FLAGS.target_vocab_size)#[batch vocabsz]
        prob_num_target=tf.nn.softmax(logits_num,axis=-1)
        loss1=-tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=y*tf.math.log(prob_num_target+eps),axis=-1))
        ##### loss2   multi label
        y=tf.reduce_sum(input_tensor=tf.one_hot(target_ids,depth=FLAGS.target_vocab_size),axis=1)[:,1:]
        # [batch numClass25 vocabsz]->[batch vocabsz] -> remove padding
        prob_y=tf.nn.sigmoid(logits_multic)[:,1:] #[batch vocabsz]
        loss2=y*tf.math.log(eps+prob_y) + (1-y)*tf.math.log(1.-prob_y+eps)
        loss2=tf.reduce_mean(input_tensor=-loss2)



        loss=loss1+loss2
        train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)




    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          #"unique_ids": unique_ids,
          #"start_logits": start_logits,
          #"end_logits": end_logits,
          "logits_multic": logits_multic,
          'logits_num':logits_num
      }
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length,
                     seq_length_y,
                     is_training,
                     drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),


  }

  if is_training:
    name_to_features['target_ids']=tf.io.FixedLenFeature([seq_length_y], tf.int64)
    name_to_features["num_of_target"]= tf.io.FixedLenFeature([], tf.int64)
    # name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    # name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.compat.v1.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.compat.v1.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.compat.v1.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.compat.v1.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.compat.v1.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.compat.v1.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
        features["target_ids"] = create_int_feature(feature.target_ids)
        features["num_of_target"] = create_int_feature([feature.num_of_target])
      # features["start_positions"] = create_int_feature([feature.start_position])
      # features["end_positions"] = create_int_feature([feature.end_position])
      # impossible = 0
      # if feature.is_impossible:
      #   impossible = 1
      # features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def body_fn(input_ids,input_mask,token_type_ids,target_ids,
            encode_model,decode_model,is_training):
    batchsz=input_ids.shape[0]
    seq_length=input_ids.shape[1]
    final_hidden = encode_model(
                                batch_size=batchsz,
                                seq_length=seq_length,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                token_type_ids=token_type_ids
                                )  # [batch step 2048]

    if is_training:
        inp = [input_ids, final_hidden, target_ids]
    else:
        inp = [input_ids, final_hidden, None]
    logits = decode_model(inp)
    return logits #[batch8 step25 vocab52]



def calc_acc(logits,target):# [batch step vocabsz] [batch step]
    logits=logits.numpy()
    target=target.numpy().flatten()
    pred=np.argmax(logits,axis=-1).flatten()# ->[batch step]
    ttcnt=0
    corr_cnt=0
    for ii in range(pred.shape[0]):
        pred_i=pred[ii]
        target_i=target[ii]
        if pred_i==0:continue  #不看padding
        #if pred_i==54:continue# 54 unlabel  # 不看unlabel的准确率
        ###
        ttcnt+=1
        if pred_i==target_i:corr_cnt+=1
    return corr_cnt/float(ttcnt+0.000001)


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  bert_config.type_vocab_size=FLAGS.pos_vocab_size ### change token_type_vocabsz


  validate_flags_or_throw(bert_config)

  tf.io.gfile.makedirs(FLAGS.output_dir)

  # tokenizer = tokenization.FullTokenizer(
  #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  #
  #
  # tokenizer_y = tokenization.FullTokenizer_word(
  #     vocab_file=FLAGS.target_vocab, do_lower_case=FLAGS.do_lower_case)###???
  # tokenizer_pos = tokenization.FullTokenizer(
  #     vocab_file=FLAGS.pos_vocab, do_lower_case=FLAGS.do_lower_case)  ###???

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))




##########  input data
  tf_filename = os.path.join(FLAGS.output_dir, "train.tf_record")
  train_input_fn = input_fn_builder(
        #input_file=train_writer.filename,
        input_file=tf_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        seq_length_y=FLAGS.max_seq_length_y)
  inp_dataset=train_input_fn({'batch_size':FLAGS.train_batch_size})


  # for step, next_ele in enumerate(inp_dataset):
  #     print ('')


  #########   初始化模型参数
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
  target_id=tf.constant([[0, 0, 1], [0, 2, 0]])





  bertEncode = modeling.BertModel(bert_config,
                                     True,
                                     batch_size=8,
                                     seq_length=FLAGS.max_seq_length,
                                     use_one_hot_embeddings=False,
                                     scope=None)

  ### restore bert encode init ckpt
  # ckpt = tf.train.Checkpoint(net=bertEncode)
  # checkpoint_path = tf2_ckpt
  # ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))



  #### decode model
  decode_model = transformerDecoder_v2.TransformerDecoder(GLOBAL_PARAMS_symptom, train=True)

  # forward
  body_fn(input_ids,
          input_mask,
          token_type_ids,
          target_id,
          bertEncode, decode_model, True)






  ### optimizer
  optimizer = keras.optimizers.SGD(learning_rate=0.001)
  optimizer=keras.optimizers.RMSprop()


  ### ckpt,manager,
  ckpt = tf.train.Checkpoint(net=bertEncode,
                                     decode=decode_model,
                                     #opt=optimizer,
                                     step=tf.Variable(1))

  manager = tf.train.CheckpointManager(
      ckpt, directory=FLAGS.output_dir, max_to_keep=5)


  ##### 恢复
  status = ckpt.restore(manager.latest_checkpoint)

  #### train iter
  is_training=True

  for step, next_ele in enumerate(inp_dataset):
      with tf.GradientTape() as tape:

          logits = body_fn(next_ele['input_ids'],
                           next_ele['input_mask'],
                           next_ele['segment_ids'],
                           next_ele['target_ids'],
                            bertEncode, decode_model, is_training)

          lossv, xentro = metrics.transformer_loss_yr(logits, next_ele["target_ids"],
                                                      GLOBAL_PARAMS_symptom["label_smoothing"],
                                                      GLOBAL_PARAMS_symptom["vocab_size"])

          if step%print_freq==0:

            acc=calc_acc(logits, next_ele['target_ids'])
            print('step', step, 'loss', lossv.numpy(),'acc no PAD',acc)  # xentro [batch step]

          xentro_flatten = tf.reshape(xentro, shape=[-1])



      weights_ = tuple(bertEncode.trainable_variables) + tuple(decode_model.trainable_variables)
      weights = [w for w in weights_ if 'pool' not in w.name]

      grads = tape.gradient(lossv, weights)
      optimizer.apply_gradients(zip(grads, weights))

      ####
      save_freq = FLAGS.save_checkpoints_steps
      if step % save_freq == 0:
          manager.save()












if __name__ == "__main__":

  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  local_flag = True
  ##
  import os

  if local_flag == True:
      OLD_BERT_MODEL_DIR = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12'
      XIAOBAI_DATA_DIR = '../bert_demo_data/knowlege/'
      FLAGS.data_dir = '../tmp/'
      FLAGS.output_dir = '../model_specifiedTask/'
      tf2_ckpt='/Users/admin/Desktop/migrate_to_tf2_bert/model_encode_ckpt'
  elif local_flag == False:
      OLD_BERT_MODEL_DIR = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12'
      XIAOBAI_DATA_DIR = '/code/bert_train_multiclass_tf2/bert_demo_data/knowlege'
      FLAGS.data_dir = '/code/bert_train_multiclass_tf2/tmp/'
      #FLAGS.output_dir = '/code/bert_train_yr_squad_v0101/model_specifiedTask/'
      FLAGS.output_dir='/models/bert_train_multiclass_tf2/'
      tf2_ckpt='/code/migrate_to_tf2_bert/model_encode_ckpt'

  FLAGS.bert_config_file = os.path.join(OLD_BERT_MODEL_DIR, 'bert_config.json')
  FLAGS.do_train = True


  FLAGS.init_checkpoint = os.path.join(OLD_BERT_MODEL_DIR, 'bert_model.ckpt')

  FLAGS.num_train_epochs = 10000
  FLAGS.learning_rate = 3e-5
  FLAGS.train_batch_size = 20
  FLAGS.label_smoothing = 0.1


  ####
  f1=['dev1.json','train1.json']
  FLAGS.train_file=os.path.join(XIAOBAI_DATA_DIR,'tmp',f1[0])
  FLAGS.vocab_file=os.path.join(OLD_BERT_MODEL_DIR,'vocab.txt')

  ### target seq


  FLAGS.target_vocab=os.path.join(XIAOBAI_DATA_DIR,'dict_v1231','spo.txt')

  FLAGS.target_vocab_size = 78
  ### pos vocab
  FLAGS.pos_vocab=os.path.join(XIAOBAI_DATA_DIR,'dict_v1231','pos.txt')

  FLAGS.pos_vocab_size=26

  FLAGS.save_checkpoints_steps = 1000

  # GLOBAL_PARAMS_target = model_params.BASE_PARAMS.copy()
  # GLOBAL_PARAMS_target["vocab_size"] = FLAGS.target_vocab_size
  # GLOBAL_PARAMS_target['num_hidden_layers']=3
  # GLOBAL_PARAMS_target['hidden_size']=128

  FLAGS.max_seq_length_y=76
  FLAGS.max_seq_length=130


  #### decode para
  GLOBAL_PARAMS_symptom = model_params.BASE_PARAMS.copy()
  GLOBAL_PARAMS_symptom["vocab_size"] = FLAGS.target_vocab_size
  GLOBAL_PARAMS_symptom['num_hidden_layers'] = 3
  GLOBAL_PARAMS_symptom['hidden_size'] = 256
  GLOBAL_PARAMS_symptom['dtype'] = tf.float32

  print_freq=100

  tf.compat.v1.app.run()
