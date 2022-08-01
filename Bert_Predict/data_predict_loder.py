# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/25
# 该模型用于处理待预测数据成Bert需要数据，更多详细内容可以参见文件Bert_Genrator.data_preprocess

import d2l.torch
import pandas as pd
import torch
from d2l import torch as d2l
import re

class data_generate():
    def __init__(self):
        # 待预测文件的位置
        self.data_file = r'./data_random_get.csv'
        # self.data_file = r'../data/test.csv'
        # self.data_file = r'../data/train2.csv'
        # self.data_file = r'../data/train.csv'
    # 函数同Bert_Genrator.data_preprocess中一致，不同的是不需要划分测试集和验证集
    def data_read(self):
        lbl=['Abdominal+Fat', 'Artificial+Intelligence', 'Culicidae',
       'Humboldt states', 'Diabetes+Mellitus', 'Fasting',
       'Gastrointestinal+Microbiome', 'Inflammation', 'MicroRNAs', 'Neoplasms',
       'Parkinson+Disease', 'psychology']
        train_data = pd.read_csv(self.data_file, sep=',')
        if 'Topic(Label)' in train_data.columns:
            y = [lbl.index(i) for i in train_data['Topic(Label)']]
        else:
            y=None
        train_data['Abstract'] = train_data['Abstract'].fillna('')
        text = []
        for i in range(len(train_data)):
            Title1 = train_data.loc[i, 'Title'].strip().replace('\n', "").replace("[", "").replace("]", "").replace("-"," ").replace(". ", " , ").lower()
            abstract = train_data.loc[i, 'Abstract']
            abstract_plus = re.sub(" {2,}|[^a-z,.:;?]{3,}", " ", re.sub(r"\(.*?\)|\[.*?]|[)(\-]", " ",
                                                                        abstract.strip().replace("\n", " ").replace(","," , ").replace(":", " : ").lower()))
            text.append((Title1 + " . " + abstract_plus.replace('!', " , ").replace('?', " , ").replace(". ", " , ").replace("; "," , ")))
        return text,lbl,y
    def paragraphs(self):
        predict_lines, lbl ,y= self.data_read()
        def exception(line):
            raise Exception(f'the lines is "{line}"')
        predict_paragraphs = [line.split('. ') if len(line.split('. ')) >= 2 else exception(line) for line in predict_lines ]
        return predict_paragraphs,lbl,y

# data_preprocess同Bert_Genrator.data_preprocess文件中一致，这里不过多赘述
class data_preprocess(torch.utils.data.Dataset):
    def __init__(self, predict_paragraphs, max_len, vocab):
        paragraphs = predict_paragraphs
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        self.vocab = vocab
        self.max_len = max_len
        self.valid_lens = []
        self.all_token_ids = []
        self.all_segments = []
        for i, paragraph in enumerate(paragraphs):
            self._get_nsp_data_from_paragraph(paragraph)

    def __getitem__(self, idx):
        content=self._pad_bert_inputs(idx)
        return content

    def __len__(self):
        return len(self.all_token_ids)

    def _pad_bert_inputs(self, idx):
        paragraph_token_ids,paragraphs_segments=self.all_token_ids[idx],self.all_segments[idx]
        all_token_ids,all_segments=[],[]
        for i in range(len(paragraph_token_ids)):
            all_token_ids.append(paragraph_token_ids[i] + [self.vocab['<pad>']] * (self.max_len - len(paragraph_token_ids[i])))
            all_segments.append(paragraphs_segments[i] + [0] * (self.max_len - len(paragraphs_segments[i])))
        return [torch.tensor(all_token_ids, dtype=torch.long), torch.tensor(all_segments, dtype=torch.long), torch.tensor(self.valid_lens[idx])]

    def _get_nsp_data_from_paragraph(self, paragraph):

        def get_tokens_and_segments(tokens_a, tokens_b=None):
            tokens = ['<cls>'] + tokens_a + ['<sep>']
            segments = [0] * (len(tokens_a) + 2)
            if tokens_b is not None:
                tokens += tokens_b + ['<sep>']
                segments += [1] * (len(tokens_b) + 1)
            return tokens, segments

        def _truncate_pair_of_tokens(tokens_a, tokens_b):
            while len(tokens_a) + len(tokens_b) > self.max_len - 3:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        each_paragraph_tokens_id,each_paragraph_segments,each_paragraph_validlen = [],[],[]
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b = paragraph[i], paragraph[i + 1]
            if len(tokens_a) + len(tokens_b) + 3 > self.max_len:
                _truncate_pair_of_tokens(tokens_a, tokens_b)
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            tokens_id = self.vocab[tokens]
            each_paragraph_tokens_id.append(tokens_id)
            each_paragraph_segments.append(segments)
            each_paragraph_validlen.append(len(tokens))
        self.all_token_ids.append(each_paragraph_tokens_id)
        self.all_segments.append(each_paragraph_segments)
        self.valid_lens.append(each_paragraph_validlen)


def load_data_data(max_len, vocab):
    Class = data_generate()
    predict_paragraphs,lbl,y = Class.paragraphs()
    predict_set = data_preprocess(predict_paragraphs, max_len, vocab)
    return predict_set,lbl,y


# 不同于Bert_Genrator.data_preprocess文件的data_iter，这里不需要进行打散处理，按顺序一条一条读取就行
def data_iter(set):
    num_examples = len(set)
    for i in range(0, num_examples):
        yield set[i]
