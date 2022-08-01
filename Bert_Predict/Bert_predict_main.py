# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/24
# 该脚本用来进行数据的预测

# sys.path.append(r'..\..\Paper_identify')
from torch import nn
import torch
from d2l import torch as d2l
from Bert_Genrator.bert_load import load_finetune_model,load_finetune_model_mmp,\
    load_finetune_model_gru
from data_predict_loder import load_data_data, data_iter
import numpy as np
from TF_IDF import data_generate


# 下面函数为本文试图通过集成学习写的脚本，但效果不太行，故没有使用，可忽略，
# 1是较为综合模型,2是困难样本模型.综合模型变化到困难样本模型，如果最大值变化率较大就选最大值索引作为正确标签，反之同理,
# shreshold_rate用来判断net1中最大值和第二大值的比例为多少判定为足够接近，shreshold是用来判断从net到net1，net1的最大值和第二大值的变化是否足够接近的阈值
def judge(y_hat_softmax,y_hat_softmax1,shreshold_rate,shreshold_rate1):

    def get_dic(softmax1, softmax2):  # softmax1,softmax2均为列表
        dic = dict()
        dic1 = dict()
        dic['max'] = (softmax1.index(sorted(softmax1, reverse=True)[0]), sorted(softmax1, reverse=True)[0])
        dic['second_max'] = (softmax1.index(sorted(softmax1, reverse=True)[1]), sorted(softmax1, reverse=True)[1])
        dic1['max'] = (softmax2.index(sorted(softmax2, reverse=True)[0]), sorted(softmax2, reverse=True)[0])
        dic1['second_max'] = (softmax2.index(sorted(softmax2, reverse=True)[1]), sorted(softmax2, reverse=True)[1])
        return dic, dic1

    # # 定义两个数字的是否相近，小于阈值则是相近
    # def approximate(num1, num2, shreshold):
    #     if abs(num1 - num2) <= shreshold:
    #         return True
    #     return False

    # ①先判断net1和net最大值索引是相同以及net1的最大值和第二大值比值是否超过了阈值(shreshold_rate)：
        # 均满足：直接返货net1最大值索引
        # 其中任何一个不满足：
            # 先判断net1最大值和对应索引在net的值之和是否大于net1的第二大值和对应索引在net的值之和乘以阈值(shreshold_rate1)：
                # 大于 ：直接返回net1最大值索引
                # 不大于：通过net1最大值和对应索引在net的值的变化率是否大于net1的第二大值和对应索引在net的值的变化率：
                        # 大于：直接返回net1最大值索引
                        # 不大于：直接返回net1第二大值索引
    dic,dic1=get_dic(y_hat_softmax[0].tolist(),y_hat_softmax1[0].tolist())
    # if dic['max'][0]==dic1['max'][0] and dic1['max'][1]>=shreshold_rate*dic1['second_max'][1]:
    if dic1['max'][1]>=0.7:
        return dic1['max'][0]
    else:
        # result=(y_hat_softmax1[0][dic1['max'][0]]+y_hat_softmax[0][dic1['max'][0]])
        # result1=shreshold_rate1*(y_hat_softmax1[0][dic1['second_max'][0]]+y_hat_softmax[0][dic1['second_max'][0]])
        if dic1['max'][1] >= shreshold_rate*dic1['second'][1]:
            return dic1['max'][0]
        else:
            result= (y_hat_softmax1[0][dic1['max'][0]]-y_hat_softmax[0][dic1['max'][0]])/y_hat_softmax[0][dic1['max'][0]]
            result1=(y_hat_softmax1[0][dic1['second_max'][0]] - y_hat_softmax[0][dic1['second_max'][0]]) / \
                    y_hat_softmax[0][dic1['second_max'][0]]
            if result>=result1:
                return dic1['max'][0]
            else:
                return dic1['second_max'][0]

# 预测主函数

def bert_predict(predict_set, net, net1, lbl,y,shreshold_rate,shreshold_rate1,devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # 将集成学习（存在多个预测模型）的其他模型传入GPU
    if net1:
        net1 = nn.DataParallel(net1, device_ids=devices).to(devices[0])
    # 生成预测数据，每次生成一条数据
    predict_iter = data_iter(predict_set)
    # 最终预测结果
    predic = []
    # 如果待预测数据存在真实标签，则记录模型预测正确的概率
    a=0
    for i, features in enumerate(predict_iter):
        # pred_id, y_prob, Loss = train_batch_ch13(net, features, devices,y,i,shreshold_rate,shreshold_rate1)
        pred_id, y_prob, Loss = train_batch_ch13(net, features, devices,y,i)
        # if prob[0] >= 0:
        prediction = lbl[pred_id[0]]
        # prediction = lbl[pred_id]
        # 将每个样本标签的预测概率进行列表话并打印
        print(f"{y_prob.tolist()}")
        if y:
            if prediction==lbl[y[i]]:
                a+=1
        # else:
        #     # 原始模型
        #     prediction0 = lbl[pred_id[0]]
        #     # 困难样本模型
        #     pred_id1, prob1 = train_batch_ch13(net1, features, devices)
        #     prediction1 = lbl[pred_id1[0]]
        #     if prob1[0] >= 0:
        #         prediction=prediction1
        #         print(f"net1:{prob1[0]:.3f}")
        #     else:
        #         # TF-IDF模型
        #         index = clf.predict(x_predict[i])[0]
        #         prediction2 = lbl_tfidf[index]
        #         prediction=vote([prediction0,prediction1,prediction2])
        #         print(f"未知:{prediction}")
        # 每个样本的预测结果
        predic.append(prediction)
    if a:
        # 如果是待预测数据具有真实标签，则计算预测精确度
        print(f"准确率：{a/(i+1)}")
        # print(f"shreshold_rate:{shreshold_rate},shreshold_rate1:{shreshold_rate1},准确率：{a/(i+1)}")
    return predic

# 单个样本的模型预测
def train_batch_ch13(net, X, devices,y,i):
    # 根据标签预测最大概率选择预测结果
    def predice_sum(y_hat_softmax):
        result = d2l.argmax(y_hat_softmax, axis=1)
        return result
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    # 样本的预测结果，【1，num_label】维,num_label为标签种类
    y_hat = torch.detach(net(X))
    # 将预测结果的sofemax计算
    y_hat_softmax = nn.functional.softmax(y_hat, dim=1)
    # 判断待预测样本是否存在真实标签，如果存在真实标签，则多返回一个该样本的交叉熵损失
    if y:
        y_true=y[i]
        Loss = -torch.log(y_hat_softmax[0][y_true])
        pred_id = predice_sum(y_hat_softmax)
        return pred_id, y_hat_softmax[0],Loss
    else:
        pred_id = predice_sum(y_hat_softmax)
        return pred_id, y_hat_softmax[0],0


if __name__ == "__main__":
    net1=None
    # 判断能够使用的设备（GPU?或CPU?）
    devices = d2l.try_all_gpus()
    # 导入经过微调后的bert的模型，名称默认可以参见load_finetune_model函数
    net,vocab,_ = load_finetune_model()
    # net,vocab,_ = load_finetune_model_gru()
    # 集成学习其他模型的导入，并指定模型文件名称
    # net1 = load_finetune_model(model_file='mlp.params')[0]
    # 每个句子对的最大成都
    max_len = 512
    # 用于集成学习的超参数，可以忽略
    shreshold_rate,shreshold_rate1=1.25,1.25
    # 测试数据导入，predict_set为待预测数据，y为标签，如果不存在则返回None
    predict_set, lbl,y = load_data_data(max_len, vocab)
    # 集成学习用到的TF-IDF模型，具体可见data_generate函数
    # clf, x_predict, lbl_tfidf = data_generate()
    # 预测函数
    predictions = bert_predict(predict_set, net, net1, lbl,y,shreshold_rate,shreshold_rate1,devices)
    # 打印正在使用的模型
    print(f"模型：{_}")
    # 将模型预测结果写入result_predict文件
    with open('result_predict.csv', 'w', encoding='utf-8') as fp:
        fp.write('Topic(Label)' + '\n')
        for i in predictions:
            fp.write(i + '\n')

