# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/30
# 该文件主要是用于将vocab中一些不常用的词汇如数字、以及一些在训练中无法使用到的词汇替换成训练数据集中使用
# 的一些医学、科学专用名词，替换规律为先将词汇表中数字全部替换，再将词汇表中从最后一个没有出现再训练集中的词汇依次
# 替换成训练数据集中出现频率由高到低的词汇
import os
import json
import re
import pandas as pd
from collections import Counter,deque

# 全部训练数据集
data_file= r'../data/train.csv'
train_df = pd.read_csv(data_file, sep=',')
data = json.load(open(os.path.join(r"", 'vocab.json')))
# 将源词汇表中单词转化为字典形式，key为单词，value为单词序号
dic = {token: idx for idx, token in enumerate(data)}
# vocab为训练数据集中全部单词
vocab=[]
for i in range(len(train_df)):
    # print(i)
    paragraphs = re.sub(r"\(.*?\)|\[.*?]|[)(\-,.?!:;]", " ",train_df.loc[i, 'Abstract'].replace('\n', " ").strip().lower())\
       + re.sub(r"\(.*?\)|\[.*?]|[)(\-,.?!:;]", " ",train_df.loc[i, 'Title'].replace('\n', " ").strip().lower())
    vocab.extend(re.sub("[^a-z]{3,}", " ",paragraphs).split())
# 记录训练数据中单词出现次数，为字典形式，key为单词，value为单词出现频率
vocab_counter=Counter(vocab)
# 在词汇表中没有出现的训练数据集单词
vocab_not_in_dic=[]
# 在训练数据集中没有出现的词汇表中单词
dic_not_in_vocab=[]
for i,j in vocab_counter.items():
    if dic.get(i,-1)==-1:
        vocab_not_in_dic.append((i,j))

vocab_not_in_dic.sort(key=lambda item:-item[1])
vocab_not_in_dic=deque(vocab_not_in_dic)


for i,j in dic.items():
    if vocab_counter.get(i,-1)==-1 and i.isdigit() and j>10:
        data[j] = vocab_not_in_dic[0][0]
        vocab_not_in_dic.popleft()
    elif vocab_counter.get(i,-1)==-1 and j>10:
        dic_not_in_vocab.append((i,j))
dic_not_in_vocab.sort(key=lambda item:-item[1])
# print(vocab_not_in_dic)
# print(dic_not_in_vocab)
i,j,item=0,0,vocab_not_in_dic[0]
while i<len(vocab_not_in_dic)-1:
    if item[1]>=5 and j<len(dic_not_in_vocab)-1:
        # print(f"{data[dic_not_in_vocab[j][1]]} to {item[0]}")
        data[dic_not_in_vocab[j][1]]=item[0]
        j += 1
    i+=1
    item=vocab_not_in_dic[i]
print(j,len(dic_not_in_vocab))
# print(data,len(data))
with open('vocab3.json', 'w', encoding='utf-8') as fp:
    json.dump(data,fp)


