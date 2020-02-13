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


def var_assign(var,tensor): # variable 赋值
    var.assign(tensor)
    return var


def from_submodel_get_weights(m): # keras 层
    return m.get_weights()

def from_submodel_get_weights1(m): #tf.Module 层
    return m.trainable_variables[0]

def from_submodel_set_weights(m,namell,varDict): # keras 层
    wll=[varDict[name] for name in namell]
    m.set_weights(wll)

def read_var_from_ckpt(init_ckpt_path):
    from tensorflow.python import pywrap_tensorflow

    #checkpoint_path = tf.train.latest_checkpoint('./tmp1/')
    checkpoint_path=init_ckpt_path
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    name_value = {}
    values = []
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        t = reader.get_tensor(key)
        print(t.shape)
        name_value[key] = t
        values.append(t)
    return name_value,values
