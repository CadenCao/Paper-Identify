# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/30
# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/14
import os

import d2l.torch
import pandas as pd
import torch
from d2l import torch as d2l
import re
from sklearn.model_selection import train_test_split
import random


class data_generate():
    def __init__(self, test_size,data_file = r'../data/train2.csv'):
        self.data_file = data_file
        self.test_size = test_size

    def data_read(self):
        lbl = ['Abdominal+Fat', 'Artificial+Intelligence', 'Culicidae',
               'Humboldt states', 'Diabetes+Mellitus', 'Fasting',
                   'Gastrointestinal+Microbiome', 'Inflammation', 'MicroRNAs', 'Neoplasms',
               'Parkinson+Disease', 'psychology']
        train_df = pd.read_csv(self.data_file, sep=',')
        train_df['Abstract'] = train_df['Abstract'].fillna('')
        text = []
        for i in range(len(train_df)):
            Title1 = train_df.loc[i, 'Title'].strip().replace('\n', "").replace("[", "").replace("]", "").replace("-", " ").replace(". ", " , ").lower()
            abstract = train_df.loc[i, 'Abstract']
            abstract_plus = re.sub(" {2,}|[^a-z,.:;?]{3,}", " ", re.sub(r"\(.*?\)|\[.*?]|[)(\-]", " ",
                                                                        abstract.strip().replace("\n", " ").replace(","," , ").replace(":", " : ").lower()))
            text.append((Title1 + " . " , abstract_plus.replace('!', " , ").replace('?', " , ").replace(". "," , ").replace("; "," , ")))

        y =[lbl.index(i) for i in train_df['Topic(Label)']]

        if self.test_size:
            x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=self.test_size,shuffle=False)
            return x_train, x_test, y_train, y_test, lbl
        else:
            return text, None, y, None,lbl


def load_data_data(test_size,valid_file=None):
    if not valid_file:
        Class = data_generate(test_size)
        x_train, x_test, y_train, y_test, lbl = Class.data_read()
        num_train_iter=len(x_train)
        return x_train, x_test, y_train, y_test, lbl,num_train_iter
    else:
        Class = data_generate(test_size,data_file=valid_file)
        x_train, x_test, y_train, y_test, lbl = Class.data_read()
        return x_train,y_train


def data_iter(train_set,batch_size):
    # tokenizer_set_x,tokenizer_set_y=tokenizer_set
    num_examples = len(train_set[1])
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield [train_set[0][i] for i in batch_indices],\
              torch.tensor([train_set[1][i] for i in batch_indices])

if __name__=="__main__":
    pass