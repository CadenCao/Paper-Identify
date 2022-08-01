# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/28
# 该文件是TF-IDF模型预测，该模型数据懒惰学习算法，不需要提前进行训练，这里将训练集和测试集一同载入（防止两者存在单词不一致的情况）
# 然后直接使用该模型对测试集进行预测


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier,LogisticRegression
import numpy as np

def data_generate():
    train_data = pd.read_csv('../data/train1.csv', sep=',')
    predict_data = pd.read_csv(r'./data_random_get.csv', sep=',')
    # predict_data=pd.read_csv('../data/train1.csv', sep=',')

    train_data['Abstract'] = train_data['Abstract'].fillna('')
    predict_data['Abstract'] = predict_data['Abstract'].fillna('')
    lbl = ['Abdominal+Fat', 'Artificial+Intelligence', 'Culicidae',
           'Humboldt states', 'Diabetes+Mellitus', 'Fasting',
           'Gastrointestinal+Microbiome', 'Inflammation', 'MicroRNAs', 'Neoplasms',
           'Parkinson+Disease', 'psychology']
    text = []
    for i in range(len(train_data)):
        Title1 = train_data.loc[i, 'Title'].replace('[', '').replace(']', '').replace('-', " ")
        abstract1 = train_data.loc[i, 'Abstract'].split('. ')
        Abstract1 = ". ".join(abstract1[:min(3, int(len(abstract1) * 0.25))] * (5) + abstract1 +
                              abstract1[max(-3, -len(abstract1)):] * (5))
        text.append(((Title1 + " . ") * len(abstract1) + Abstract1.replace('!', " . ").replace('?', " . ")).lower())

    for i in range(len(predict_data)):
        Title1 = predict_data.loc[i, 'Title'].replace('[', '').replace(']', '').replace('-', " ")
        abstract1 = predict_data.loc[i, 'Abstract'].split('. ')
        Abstract1 = ". ".join(abstract1[:min(3, int(len(abstract1) * 0.25))] * (5) + abstract1 +
                              abstract1[max(-3, -len(abstract1)):] * (5))
        text.append(((Title1 + " . ") * len(abstract1) + Abstract1.replace('!', " . ").replace('?', " . ")).lower())
    y = [lbl.index(i) for i in train_data['Topic(Label)']]
    # TF-IDF模型生成
    tfidf = TfidfVectorizer(norm='l2', max_features=15000)
    # 计算text的tf-idf值
    X = tfidf.fit_transform(text)
    x_train,y_train=X[:len(train_data)],y
    x_predict=X[len(train_data):]
    # 这里使用随机梯度下降，损失函数为modified_huber，默认为hunge loss
    clf = SGDClassifier(loss='modified_huber')
    # x,y进行拟合
    clf.fit(x_train, y_train)
    return clf,x_predict,lbl


if __name__=="__main__":
    clf,x_predict,lbl=data_generate()
    np.set_printoptions(linewidth=200, threshold=np.inf)
    for i in x_predict:
        print(clf.predict_proba(i)[0])
