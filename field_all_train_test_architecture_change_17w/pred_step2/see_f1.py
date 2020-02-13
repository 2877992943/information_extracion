#! -*- coding:utf-8 -*-



from problem_util_yr.loadDict.read_json_tool import read_json

X, Y, Z = 1e-10, 1e-10, 1e-10
gene=read_json('step2_rst.json')

for d in gene:
    print ('')
    R=set([' '.join(spo) for spo in d['pred']])
    T=set([' '.join(spo) for spo in d['spo_list']])
    print ('text',d['text'])
    print ('pred',R)
    print ('y',T)

    X += len(R & T)
    Y += len(R)
    Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

####
print (f1,precision,recall)
