# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/30
# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/13
import json
from d2l import torch as d2l
import torch
import os
from torch import nn
from scibert_fine_tune import load_pretrained_model
import pandas as pd
import re
import transformers


class Model(nn.Module):
    def __init__(self,bert):
        super(Model,self).__init__()
        self.bert=bert
        self.output=torch.nn.Linear(768,12)
    def forward(self,inputs):
        input_ids,attention_mask, token_type_ids=inputs["input_ids"],inputs["attention_mask"],inputs["token_type_ids"]
        pooler=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[1]
        outputs=self.output(pooler)
        return outputs
def sci_load_finetune_model(model_file='mlp.params'):
    bert = load_pretrained_model()
    net = Model(bert)
    checkpoint = torch.load(os.path.join(r"./", model_file))
    net.load_state_dict(checkpoint)
    return net,model_file

class data_generate():
    def __init__(self):
        self.data_file = r'../Bert_Predict/data_random_get.csv'
        # self.data_file = r'../data/test.csv'
        # self.data_file = r'../data/train2.csv'
        # self.data_file = r'../data/train.csv'
    def data_read(self):
        lbl = ['Abdominal+Fat', 'Artificial+Intelligence', 'Culicidae',
               'Humboldt states', 'Diabetes+Mellitus', 'Fasting',
                   'Gastrointestinal+Microbiome', 'Inflammation', 'MicroRNAs', 'Neoplasms',
               'Parkinson+Disease', 'psychology']
        predict_data = pd.read_csv(self.data_file, sep=',')
        predict_data['Abstract'] = predict_data['Abstract'].fillna('')
        # 将摘要中的title列和abstract列进行合并成一个段落
        text = []
        if 'Topic(Label)' in predict_data.columns:
            y = [lbl.index(i) for i in predict_data['Topic(Label)']]
        else:
            y=None
        for i in range(len(predict_data)):
            Title1 = predict_data.loc[i, 'Title'].strip().replace('\n', "").replace("[", "").replace("]", "").replace("-", " ").replace(". ", " , ").lower()
            abstract = predict_data.loc[i, 'Abstract']
            abstract_plus = re.sub(" {2,}|[^a-z,.:;?]{3,}", " ", re.sub(r"\(.*?\)|\[.*?]|[)(\-]", " ",
                                                                        abstract.strip().replace("\n", " ").replace(","," , ").replace(":", " : ").lower()))

            text.append((Title1 + " . " , abstract_plus.replace('!', " , ").replace('?', " , ").replace(". "," , ").replace("; "," , ")))
        return text,lbl,y
def tokenizer_generate(set):
    return tokenizer.batch_encode_plus(batch_text_or_text_pairs=set,
                                                    truncation_strategy="longest_first",
                                                    max_length=512,
                                                    padding='max_length',
                                                    return_tensors="pt")


def bert_predict(predict_set,lbl,y,net,devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    predic = []
    a=0
    for i, features in enumerate(predict_set):
        features=tokenizer_generate([features])
        pred_id, y_prob, Loss = train_batch_ch13(net, features,y,i, devices)
        prediction = lbl[pred_id[0]]
        print(f"{y_prob.tolist()}")
        if y:
            if prediction==lbl[y[i]]:
                a+=1
        predic.append(prediction)
    if a:
        print(f"准确率：{a/(i+1)}")
    return predic
def train_batch_ch13(net, X, y,i,devices):
    def predice_sum(y_hat_softmax):
        result = d2l.argmax(y_hat_softmax, axis=1)
        return result
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y_hat = torch.detach(net(X))
    y_hat_softmax = nn.functional.softmax(y_hat, dim=1)
    if y:
        y_true=y[i]
        Loss = -torch.log(y_hat_softmax[0][y_true])
        pred_id = predice_sum(y_hat_softmax)
        return pred_id, y_hat_softmax[0],Loss
    else:
        pred_id = predice_sum(y_hat_softmax)
        return pred_id, y_hat_softmax[0],0

if __name__ == "__main__":
    devices = d2l.try_all_gpus()
    net,_ = sci_load_finetune_model()
    Class=data_generate()
    tokenizer = transformers.BertTokenizer.from_pretrained(r"./vocab.txt")
    predict_set, lbl, y = Class.data_read()
    predictions = bert_predict(predict_set,lbl,y,net,devices)
    print(f"模型：{_}")
    with open('sci_result_predict.csv', 'w', encoding='utf-8') as fp:
        fp.write('Topic(Label)' + '\n')
        for i in predictions:
            fp.write(i + '\n')




















