#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.82198
# 换用RoBERTa Large可以达到f1=0.829+


path_to_be_add = '../../bert4keras-master'
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

load_bert_flag = False
maxlen = 128
batch_size = 32

config_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/admin/Desktop/previous/bert2019/download_model_cn/chinese_L-12_H-768_A-12/vocab.txt'

#config_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '/code/bert_download/download_model_cn/chinese_L-12_H-768_A-12/vocab.txt'


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
#train_data = load_data('../spo_data/field1.json')
valid_data = load_data('../spo_data/dev1.json')
predicate2id, id2predicate = {}, {}

with codecs.open('../spo_data/all_50_schemas') as f:
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
        batch_p_id = []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for i in idxs:
            d = self.data[i]
            token_ids, segment_ids = tokenizer.encode(d['text'], max_length=maxlen)
            # 整理三元组 {s: [(o, p)]}
            # new s p -> o
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1)
                    # o = (o_idx, o_idx + len(o) - 1, p)# [o-start,o-end,predicate]
                    if tuple([s, p]) not in spoes:
                        spoes[tuple([s, p])] = []
                    spoes[tuple([s, p])].append(o)
            if spoes:
                # subject标签
                # subject_labels = np.zeros((len(token_ids), 2))#[69step 2]
                # object_labels = np.zeros((len(token_ids), 2))
                # subject_ids=[]
                # p_label=None

                for s, p in spoes:  # 1 text 多个 [s p] | 1个 [s,p] 有多个obj
                    # each [s p] is 1 obs
                    object_labels = np.zeros((len(token_ids), 2))
                    subject_ids = s  # [start end]
                    p_label = p

                    # subject_labels[s[0], 0] = 1
                    # subject_labels[s[1], 1] = 1

                    for o in spoes[tuple([s, p])]:
                        object_labels[o[0], 0] = 1
                        object_labels[o[1], 1] = 1

                    # 随机选一个subject
                    # start, end = np.array(list(spoes.keys())).T
                    # start = np.random.choice(start)
                    # end = np.random.choice(end[end >= start])
                    # subject_ids = (start, end)
                    # 对应的object标签
                    # object_labels = np.zeros((len(token_ids), len(predicate2id), 2)) #[69step 49predicate 2]
                    # object_labels = np.zeros((len(token_ids),  2))
                    # if len(spoes.get(subject_ids, []))>1:
                    #     print ('')
                    # for o in spoes.get(subject_ids, []):
                    #     object_labels[o[0], o[2], 0] = 1
                    #     object_labels[o[1], o[2], 1] = 1
                    # 构建batch
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    # batch_subject_labels.append(subject_labels)
                    batch_subject_ids.append(subject_ids)
                    batch_object_labels.append(object_labels)
                    batch_p_id.append([p_label])
                    if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)
                        # batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                        batch_p_id = np.array(batch_p_id)
                        batch_subject_ids = np.array(batch_subject_ids)
                        batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((2)))
                        yield [
                                  batch_token_ids,  # [batch step]
                                  batch_segment_ids,
                                  # batch_subject_labels,
                                  batch_subject_ids,  # [batch 2]
                                  batch_p_id,  # [batch,1]
                                  batch_object_labels  # [batch step 2]
                              ], None
                        batch_token_ids, batch_segment_ids = [], []
                        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
                        batch_p_id = []


def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = batch_gather(output, subject_ids[:, :1])  # [? ? 768]->[? 1 768]
    end = batch_gather(output, subject_ids[:, 1:])  # [? 1 768]
    subject = K.concatenate([start, end], 2)  # [? 1 1536]
    return subject[:, 0]


def concat(xll):
    return K.concatenate([xll[0], xll[1][:, 0, :]], -1)


#### 测试 生成数据
# train_generator = data_generator(train_data, batch_size)
#
# for ite in train_generator:
#     print ('')


# 补充输入
# subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
# object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')
object_labels = Input(shape=(None, 2), name='Object-Labels')  # [batch step 2]
# predicate_id=tf.keras.Input(shape=(),name='predicate')#[batch 1]
# predicate_id=tf.placeholder(shape=[None],dtype=tf.int32) # not work
predicate_id = Input(shape=(1,), name='predicateid')

# 加载预训练模型
checkpoint_path1 = None if load_bert_flag == False else checkpoint_path
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path1,
    return_keras_model=False,
)

# 预测subject
# output = Dense(units=2,
# activation='sigmoid',
# kernel_initializer=bert.initializer)(bert.model.output) #[? ? 768]->[? ? 2]
# subject_preds = Lambda(lambda x: x**2)(output)

# subject_model = Model(bert.model.inputs, subject_preds) # text -> sub

# 传入subject，预测object
# 传入 predicate
# 通过Conditional Layer Normalization将subject融入到object的预测中

## embed predicate
predicate_n = 49
emb_l = Embedding(predicate_n, 32, name='p_emb')  # not tf.keras.layer, | keras.layer.xx
predicate_emb = emb_l(predicate_id)

output = bert.model.layers[-2].get_output_at(-1)
subject = Lambda(extrac_subject)([output, subject_ids])  # output[? ? 768]  subid[? 2]  ->subject[? 768*2]

###
# subject_predicate=K.concatenate([subject,predicate_emb[:,0,:]],axis=-1)
# subject_predicate=Concatenate(-1)([subject,predicate_emb[:,0,:]])
subject_predicate = Lambda(concat)([subject, predicate_emb])
output = LayerNormalization(conditional=True, name='specialNorm')(
    [output, subject_predicate])  # [? ? 768] mean std 依赖sub-hid
output = Dense(units=2,
               # units=len(predicate2id) * 2,
               activation='sigmoid',
               kernel_initializer=bert.initializer)(output)
# output = Reshape((-1, 2))(output) #[? ? 2]
object_preds = Lambda(lambda x: x ** 4)(output)

object_model = Model(bert.model.inputs + [subject_ids, predicate_id], object_preds)  # sub,text -> obj,predicate

# 训练模型
# train_model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
#                     [subject_preds, object_preds])
train_model = Model(bert.model.inputs + [subject_ids, predicate_id, object_labels],
                    [object_preds])

mask = bert.model.get_layer('Sequence-Mask').output_mask

# subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
# subject_loss = K.mean(subject_loss, 2)
# subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

object_loss = K.binary_crossentropy(object_labels, object_preds)  # [batch step 2]
object_loss = K.mean(object_loss, 2)
object_loss = K.sum(object_loss * mask) / K.sum(mask)

train_model.add_loss(object_loss)
train_model.compile(optimizer=Adam(1e-5))

#
# def extract_spoes(text):
#     # 抽取输入text所包含的三元组
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

##训练用的
def extract_o_given_sp_for_training(d):  # 1 text
    # 抽取输入text所包含的三元组
    text = d['text']
    tokens = tokenizer.tokenize(text, max_length=maxlen)
    # token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # 整理三元组 {s: [(o, p)]}
    # new s p -> o
    spoes = {}  #

    for s, p, o in d['spo_list']:
        s = tokenizer.encode(s)[0][1:-1]
        p = predicate2id[p]
        o = tokenizer.encode(o)[0][1:-1]
        s_idx = search(s, token_ids)
        o_idx = search(o, token_ids)
        if s_idx != -1 and o_idx != -1:
            s = (s_idx, s_idx + len(s) - 1)
            o = (o_idx, o_idx + len(o) - 1)
            # o = (o_idx, o_idx + len(o) - 1, p)# [o-start,o-end,predicate]
            if tuple([s, p]) not in spoes:
                spoes[tuple([s, p])] = []
            spoes[tuple([s, p])].append(o)



    ###
    spos_ll = []  # return
    ##### 收集到 所有  [s,p] -> obj,obj....
    for sp, os in spoes.items():
        s = np.array([sp[0]])  # [batch 2]
        p = np.array([[sp[1]]])  # [batch 1]
        obj_preds = object_model.predict([[token_ids], [segment_ids], s, p])  # o[batch step 2]

        ####
        start = np.where(obj_preds[0, :, 0] > 0.6)[0]
        end = np.where(obj_preds[0, :, 1] > 0.5)[0]
        objs = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                objs.append((i, j))
        ###

        s_str = tokenizer.decode(token_ids[sp[0][0]:sp[0][1] + 1], tokens[sp[0][0]:sp[0][1] + 1])
        p_str = id2predicate[sp[1]]
        for o in objs:
            o_str = tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            #
            spos_ll.append([s_str, p_str, o_str])
        ####
    #####
    return spos_ll


#######  接着 step1 预测用的
def extract_o_given_sp(d):  # 1 text
    # 抽取输入text所包含的三元组
    text = d['text']
    tokens = tokenizer.tokenize(text, max_length=maxlen)
    # token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
    # 整理三元组 {s: [(o, p)]}
    # new s p -> o
    spoes = {}  #

    #for s, p, o in d['spo_list']:
    for s,p in d['s_p']:
        s = tokenizer.encode(s)[0][1:-1]
        p = predicate2id[p]
        #o = tokenizer.encode(o)[0][1:-1]
        s_idx = search(s, token_ids)
        #o_idx = search(o, token_ids)
        #if s_idx != -1 and o_idx != -1:
        if s_idx != -1 :
            s = (s_idx, s_idx + len(s) - 1)
            #o = (o_idx, o_idx + len(o) - 1)
            # o = (o_idx, o_idx + len(o) - 1, p)# [o-start,o-end,predicate]
            if tuple([s, p]) not in spoes:
                spoes[tuple([s, p])] = []
            #spoes[tuple([s, p])].append(o)



    ###
    spos_ll = []  # return
    ##### 收集到 所有  [s,p] -> obj,obj....
    #for sp, os in spoes.items():
    for sp, _ in spoes.items():
        s = np.array([sp[0]])  # [batch 2]
        p = np.array([[sp[1]]])  # [batch 1]
        obj_preds = object_model.predict([[token_ids], [segment_ids], s, p])  # o[batch step 2]

        ####
        start = np.where(obj_preds[0, :, 0] > 0.6)[0]
        end = np.where(obj_preds[0, :, 1] > 0.5)[0]
        objs = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                objs.append((i, j))
        ###

        s_str = tokenizer.decode(token_ids[sp[0][0]:sp[0][1] + 1], tokens[sp[0][0]:sp[0][1] + 1])
        p_str = id2predicate[sp[1]]
        for o in objs:
            o_str = tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            #
            spos_ll.append([s_str, p_str, o_str])
        ####
    #####
    return spos_ll


    ### get s p
    # # 抽取subject
    # subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    # start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    # end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    # subjects = []
    # for i in start:
    #     j = end[end >= i]
    #     if len(j) > 0:
    #         j = j[0]
    #         subjects.append((i, j))
    # if subjects:
    #     spoes = []
    #     token_ids = np.repeat([token_ids], len(subjects), 0)
    #     segment_ids = np.repeat([segment_ids], len(subjects), 0)
    #     subjects = np.array(subjects)
    # 传入subject，抽取object和predicate
    #     object_preds = object_model.predict([token_ids, segment_ids, subjects])
    #     for subject, object_pred in zip(subjects, object_preds):
    #         start = np.where(object_pred[:, :, 0] > 0.6)
    #         end = np.where(object_pred[:, :, 1] > 0.5)
    #         for _start, predicate1 in zip(*start):
    #             for _end, predicate2 in zip(*end):
    #                 if _start <= _end and predicate1 == predicate2:
    #                     spoes.append((subject, predicate1, (_start, _end)))
    #                     break
    #     return [
    #         (
    #             tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
    #             id2predicate[p],
    #             tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
    #         ) for s, p, o in spoes
    #     ]
    # else:
    #     return []


class SPO(tuple):
    # 用来存三元组的类
    # 表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    # 使得在判断两个三元组是否等价时容错性更好。

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


def evaluate(data):
    # 评估函数，计算f1、precision、recall

    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = codecs.open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        # R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        # T = set([SPO(spo) for spo in d['spo_list']])
        R = set([' '.join(spo) for spo in extract_o_given_sp(d)])
        T = set([' '.join(spo) for spo in d['spo_list']])
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
    # 评估和保存模型

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




    train_model.load_weights('./best_model2.weights')

    from problem_util_yr.loadDict.read_json_tool import read_json

    gene=read_json('../pred_step1/step1_rst.json')
    writer=open('step2_rst.json','w')
    for d in gene:
        spo=extract_o_given_sp(d)
        #print ('')
        d['pred']=spo
        writer.write(json.dumps(d,ensure_ascii=False)+'\n')


    #rst = evaluate(valid_data)
    #print(rst)

    #rst = evaluate(train_data)
    #print(rst)


