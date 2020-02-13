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

import numpy as np


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



def calc_acc_multiclass(logits,target):# [batch step vocabsz] [batch step]
    logits=logits.numpy()
    target=target.numpy()
    pred=np.argmax(logits,axis=-1) # ->[batch step]
    ttcnt=0
    corr_cnt=0
    #
    ttcnt1=0
    corr_cnt1=0
    for ii in range(pred.shape[0]): # each obs [step,]
        pred_i=pred[ii]
        target_span=target[ii]


        ###  预测的 在target 里面
        for pred_ij in pred_i:
            if pred_ij == 0: continue  # 不看padding
            ttcnt+=1
            if pred_ij in target_span:corr_cnt+=1
        #### target 在预测的里面
        for target_ij in target_span:
            if target_ij==0:continue# 不看padding
            ttcnt1+=1
            if target_ij in pred_i:corr_cnt1+=1

    return corr_cnt/float(ttcnt+0.000001),corr_cnt1/float(ttcnt1+0.000001)