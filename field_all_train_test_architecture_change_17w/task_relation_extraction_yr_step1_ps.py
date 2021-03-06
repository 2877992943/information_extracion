#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.82198
# 换用RoBERTa Large可以达到f1=0.829+


# 原始的是 text -> s 1个 #[batch step 2]
# text, s -> obj,p 多个  #[batch step predicate 2]

## 修改
# text -> s 多个 [batch step 1]
# text,s(1个) -> obj p 多个[batch step predicate 2]



path_to_be_add='/code/bert4keras/bert4keras-master'
import sys
from sys import path

sys.path.insert(0, path_to_be_add)

import json
import codecs
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, ExponentialMovingAverage
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import *
from keras.models import Model
from tqdm import tqdm
import os





maxlen = 128
batch_size = 32

config_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/vocab.txt'



config_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/vocab.txt'



def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [
                    (spo['subject'], spo['predicate'], spo['object'])
                    for spo in l['spo_list']
                ]
            })
    return D


# 加载数据集
# train_data = load_data('../spo_data/train1.json')
# valid_data = load_data('../spo_data/dev1.json')
filep='/code/field_all_train_test_architecture_change_17w/spo_data'
train_data = load_data(os.path.join(filep,'train1.json'))
valid_data = load_data(os.path.join(filep,'dev1.json'))
predicate2id, id2predicate = {}, {}

with codecs.open(os.path.join(filep,'all_50_schemas')) as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for i in idxs:
            d = self.data[i]
            token_ids, segment_ids = tokenizer.encode(d['text'], max_length=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)# [o-start,o-end,predicate]
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                #subject_labels = np.zeros((len(token_ids),2))
                subject_labels = np.zeros((len(token_ids), len(predicate2id),2))#[69step ,49,2]
                for s in spoes:
                    for o_s,o_e,p in spoes[s]:
                        subject_labels[s[0], p,0] = 1
                        subject_labels[s[1], p,1] = 1
                #



                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                #batch_subject_ids.append(subject_ids)
                #batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels,
                                                            #padding=np.zeros(2),
                                                            padding=np.zeros((len(predicate2id), 2))
                                                            )
                    #batch_subject_ids = np.array(batch_subject_ids)
                    #batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(predicate2id), 2)))
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels,
                        #batch_subject_ids, batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = batch_gather(output, subject_ids[:, :1]) #[? ? 768]->[? 1 768]
    end = batch_gather(output, subject_ids[:, 1:]) #[? 1 768]
    subject = K.concatenate([start, end], 2)  #[? 1 1536]
    return subject[:, 0]





train_generator = data_generator(train_data, batch_size)

### 测试数据生成
# for ite in train_generator:
#     print ('')







# 补充输入
subject_labels = Input(shape=(None,len(predicate2id),2), name='Subject-Labels')
#subject_ids = Input(shape=(2, ), name='Subject-Ids')
#object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
#n_u=2
n_u=len(predicate2id) * 2
output = Dense(units=n_u,
               activation='sigmoid',
               kernel_initializer=bert.initializer)(bert.model.output) #[? ? 768]->[? ? 2*49]
subject_preds = Lambda(lambda x: x**2)(output)
subject_preds=Reshape((-1, len(predicate2id), 2))(subject_preds) #[? ? 49 2]

subject_model = Model(bert.model.inputs, subject_preds) # text -> sub_predicate 多个

#
# # 传入subject，预测object
# # 通过Conditional Layer Normalization将subject融入到object的预测中
# output = bert.model.layers[-2].get_output_at(-1)
# subject = Lambda(extrac_subject)([output, subject_ids]) # output[? ? 768]  subid[? 2]  ->[? 768*2]
# output = LayerNormalization(conditional=True)([output, subject]) #[? ? 768] mean std 依赖sub-hid
# output = Dense(units=len(predicate2id) * 2,
#                activation='sigmoid',
#                kernel_initializer=bert.initializer)(output)
# output = Reshape((-1, len(predicate2id), 2))(output) #[? ? predicate49*2]->[? ? 49 2]
# object_preds = Lambda(lambda x: x**4)(output)
#
# object_model = Model(bert.model.inputs + [subject_ids], object_preds) #sub,text -> obj,predicate

# 训练模型
# train_model = Model(bert.model.inputs + [subject_labels,
#                                         subject_ids, object_labels],
#                     [subject_preds, object_preds])
train_model = Model(bert.model.inputs + [subject_labels],
                    [subject_preds])

mask = bert.model.get_layer('Sequence-Mask').output_mask

subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
subject_loss = K.sum(K.mean(subject_loss, 3), 2)
subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

#
# object_loss = K.binary_crossentropy(object_labels, object_preds)
# object_loss = K.sum(K.mean(object_loss, 3), 2)
# object_loss = K.sum(object_loss * mask) / K.sum(mask)

train_model.add_loss(subject_loss )
train_model.compile(optimizer=Adam(1e-5))

#
# def extract_spoes(text):
#     #抽取输入text所包含的三元组
#
#     tokens = tokenizer.tokenize(text, max_length=maxlen)
#     token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
#     # 抽取subject
#     subject_preds = subject_model.predict([[token_ids], [segment_ids]])
#     start = np.where(subject_preds[0, :, 0] > 0.6)[0]
#     end = np.where(subject_preds[0, :, 1] > 0.5)[0]
#     subjects = []
#     for i in start:
#         j = end[end >= i]
#         if len(j) > 0:
#             j = j[0]
#             subjects.append((i, j))
#     if subjects:
#         spoes = []
#         token_ids = np.repeat([token_ids], len(subjects), 0)
#         segment_ids = np.repeat([segment_ids], len(subjects), 0)
#         subjects = np.array(subjects)
#         # 传入subject，抽取object和predicate
#         object_preds = object_model.predict([token_ids, segment_ids, subjects])
#         for subject, object_pred in zip(subjects, object_preds):
#             start = np.where(object_pred[:, :, 0] > 0.6)
#             end = np.where(object_pred[:, :, 1] > 0.5)
#             for _start, predicate1 in zip(*start):
#                 for _end, predicate2 in zip(*end):
#                     if _start <= _end and predicate1 == predicate2:
#                         spoes.append((subject, predicate1, (_start, _end)))
#                         break
#         return [
#             (
#                 tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
#                 id2predicate[p],
#                 tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
#             ) for s, p, o in spoes
#         ]
#     else:
#         return []


def extract_spoes(text):
    # 抽取输入text所包含的三元组

    tokens = tokenizer.tokenize(text, max_length=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    # start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    # end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    start = np.where(subject_preds[0, :, :, 0] > 0.6)  # [0] # [batch step predicate 2]
    end = np.where(subject_preds[0, :, :, 1] > 0.5)  # [0]
    subjects = []
    p_subs = []
    for _start, predicate1 in zip(*start):
        for _end, predicate2 in zip(*end):
            if _start <= _end and predicate1 == predicate2:
                p_subs.append((predicate1, (_start, _end)))

    if p_subs:
        ret = []
        for p, s in p_subs:
            s0, s1 = s
            # r1=tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
            r1 = tokenizer.decode(token_ids[s0:s1 + 1], tokens[s0:s1 + 1]),
            r2 = id2predicate[p]
            ret.append([r1[0], r2])  # s,p
        return ret


    else:
        return []



class SPO(tuple):
    #用来存三元组的类 表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法， 使得在判断两个三元组是否等价时容错性更好。
    
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

# testname=  'dev_pred.json'
# def evaluate(data):
#     #评估函数，计算f1、precision、recall
#
#     X, Y, Z = 1e-10, 1e-10, 1e-10
#     f = codecs.open(testname, 'w', encoding='utf-8')
#     pbar = tqdm()
#     for d in data:
#         R = set([SPO(spo) for spo in extract_spoes(d['text'])])
#         T = set([SPO(spo) for spo in d['spo_list']])
#         X += len(R & T)
#         Y += len(R)
#         Z += len(T)
#         f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
#         pbar.update()
#         pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
#                              (f1, precision, recall))
#         s = json.dumps(
#             {
#                 'text': d['text'],
#                 'spo_list': list(T),
#                 'spo_list_pred': list(R),
#                 'new': list(R - T),
#                 'lack': list(T - R),
#             },
#             ensure_ascii=False,
#             indent=4)
#         f.write(s + '\n')
#     pbar.close()
#     f.close()
#     return f1, precision, recall


def evaluate(data):
    # 评估函数，计算f1、precision、recall

    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = codecs.open('dev_pred1.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        # R = set([SPO1(spo) for spo in extract_spoes(d['text'])])
        # T = set([SPO1(spo) for spo in d['spo_list']])
        R = set([s + ' ' + p for s, p in extract_spoes(d['text'])])
        T = set([s + ' ' + p for s, p, o in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = json.dumps(
            {
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    #评估和保存模型
     
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('best_model.weights')
        EMAer.reset_old_weights()
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
              (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':



    train_generator = data_generator(train_data, batch_size)

    # for ite in train_generator:
    #     print ('')


    evaluator = Evaluator()
    EMAer = ExponentialMovingAverage(0.999)

    train_model.fit_generator(train_generator.forfit(),
                             steps_per_epoch=len(train_generator),
                             epochs=20,
                             #callbacks=[evaluator, EMAer]
                              )
    ##
    train_model.save_weights('/models/0116/pred_step1/best_model1.weights')
    #f1: 0.92734, precision: 0.89933, recall: 0.95714

# else:
#
#     train_model.load_weights('best_model.weights')



    #evaluate()
    ### evaluate
    datall = load_data(valid_data)
    rst = evaluate(datall)
    print(rst)

