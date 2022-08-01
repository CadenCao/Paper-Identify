# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/24
# 该脚本是用于将样本生成为所需数据，数据原始格式为一个数据框，包括Title，Abstract，Topic(Label)等六列，数据框每一行为一个样本数据，其中每行的
# Title、Abstract变成一个句子对（列表格式，里面全是512长度的数字），而每行的Topic(Label)变成一个数字，代表标签对应的数字。
# 最后需要的数据为两个文件：一个是X，其中X包括了3个张量，分别为all_token_ids,all_segments,valid_lens,其中all_token_ids
# 记录的每个样本经过vocab后的数字化编码，维度为【batch_size,max_len】，其中max_len为了每个句子对的最大长度（512），all_segments
# 记录了all_token_ids中每个单词所属的句子，维度同为【batch_size,max_len】，valid_lens记录了每个句子对的有效长度（也即不包括padding的长度）
# 维度为【batch_size】
import os
import d2l.torch
import pandas as pd
import torch
from d2l import torch as d2l
import re
from sklearn.model_selection import train_test_split
import random


# 将数据框格式的数据变成列表格式，列表中每个元素为一个列表，记录的每个样本的分词列表
class data_generate():
    # 待训练数据
    def __init__(self, test_size,data_file = r'../data/train2.csv'):
        self.data_file = data_file
        self.test_size = test_size
    # 返回处理成段落形式的数据以及数字话后的标签，数据为列表格式，每个元素一个段落（由数据框中Title和Abstract合并而成）
    def data_read(self):
        # 数据框中标签
        lbl = ['Abdominal+Fat', 'Artificial+Intelligence', 'Culicidae',
               'Humboldt states', 'Diabetes+Mellitus', 'Fasting',
                   'Gastrointestinal+Microbiome', 'Inflammation', 'MicroRNAs', 'Neoplasms',
               'Parkinson+Disease', 'psychology']
        train_df = pd.read_csv(self.data_file, sep=',')
        # 如果数据长度为16000（全部数据长度），则将其中任意1500条转化成测试集用以评估模型的优略，测试数据不参与训练，仅仅用作评估
        if len(train_df)==16500:
            # 通过固定随机种子，使得每次生成的测试集固定
            random.seed(42)
            indices = list(range(16500))
            # 打乱indices
            random.shuffle(indices)
            # 15000条数据用以训练，1500用以测试，并生成相应的文件
            if "data_random_get.csv" not in os.listdir('../Bert_Predict'):
                train_df.iloc[indices[15000:],:].to_csv('../Bert_Predict/data_random_get.csv', index=False)
            train_df=train_df.iloc[indices[:15000],:]
            if "train1.csv" not in os.listdir('../data'):
                train_df.to_csv('../data/train1.csv', index=False)
        # 处理空白字段
        train_df['Abstract'] = train_df['Abstract'].fillna('')
        # 将摘要中的title列和abstract列进行合并成一个段落
        text = []
        for i in range(len(train_df)):
            # 对数据进行处理（包括去掉（），[]中内容，。！？：转,，未知格式的字符删除等等）
            Title1 = train_df.loc[i, 'Title'].strip().replace('\n', "").replace("[", "").replace("]", "").replace("-", " ").replace(". ", " , ").lower()
            abstract = train_df.loc[i, 'Abstract']
            abstract_plus = re.sub(" {2,}|[^a-z,.:;?]{3,}", " ", re.sub(r"\(.*?\)|\[.*?]|[)(\-]", " ",
                                                                        abstract.strip().replace("\n", " ").replace(","," , ").replace(":", " : ").lower()))
            # 将标题和摘要合并成一个段落
            text.append((Title1 + " . " + abstract_plus.replace('!', " , ").replace('?', " , ").replace(". "," , ").replace("; "," , ")))
        # 将标签进行数字化
        y =[lbl.index(i) for i in train_df['Topic(Label)']]
        # 如果self.test_size为0，表明为测试数据（全部数据用来测试），因此不需要在划分训练集和验证集，self.test_size不为零则代表
        # 是训练数据，需要划分为训练集和测试集
        if self.test_size:
            x_train, x_test, y_train, y_test = train_test_split(text, y, test_size=self.test_size)
            return x_train, x_test, y_train, y_test, lbl
        else:
            return text, None, y, None,lbl
    # 将每一段落变成一个一个由单词、标点组成的列表
    def paragraphs(self):
        train_lines, test_lines, train_labels, test_labels, lbl = self.data_read()
        def exception(line):
            raise Exception(f'the lines is "{line}"')
        train_paragraphs = [line.split('. ') if len(line.split('. ')) >= 2 else exception(line) for line in train_lines ]
        train_paragraphs_and_labels = (train_paragraphs, train_labels)
        # 测试数据没有训练集和验证集
        if test_lines:
            test_paragraphs =[line.split('. ') if len(line.split('. ')) >= 2 else exception(line) for line in test_lines ]
            test_paragraphs_and_labels = (test_paragraphs, test_labels)
        else:
            test_paragraphs_and_labels=None
        # 输出[a,b]，其中a是段落数据，b每个段落中句子数量，每个句子是一个字符串
        return train_paragraphs_and_labels, test_paragraphs_and_labels, lbl

# 将数据处理成bert模型需要输入的数据，即将样本转化为all_token_ids,all_segments,valid_lens数据集
class data_preprocess(torch.utils.data.Dataset):
    def __init__(self, paragraphs_and_labels, max_len, vocab):
        paragraphs = paragraphs_and_labels[0]
        # X，即每个样本中每个元素是由单词组成的列表
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        self.vocab = vocab
        self.max_len = max_len
        self.labels = []
        self.valid_lens = []
        self.all_token_ids = []
        self.all_segments = []
        # 将每一个由单词组成的列表插入self.all_token_ids（每个单词为数字化）【batch_size,句子对中单词数量】
        # 、self.all_segments（每个单词隶属句子，取值0或1）【batch_size,句子对中单词数量】
        # 、self.labels（每个句子对所属标签）【batch_size】
        for i, paragraph in enumerate(paragraphs):
            nums = self._get_nsp_data_from_paragraph(paragraph)
            self.labels.extend([paragraphs_and_labels[1][i]] * nums)

    def __getitem__(self, idx):
        # padding等到需要的时候在padding
        return self._pad_bert_inputs(idx), torch.tensor(self.labels)[idx]

    def __len__(self):
        return len(self.all_token_ids)

    # 数据padding任务，all_token_ids、all_segments、valid_lens均padding到【batch_size,max_len】
    def _pad_bert_inputs(self, idx):
        if not isinstance(idx, (list, tuple)):
            all_token_ids = self.all_token_ids[idx] + [self.vocab['<pad>']] * (
                        self.max_len - len(self.all_token_ids[idx]))
            all_segments = self.all_segments[idx] + [0] * (self.max_len - len(self.all_segments[idx]))
            return [torch.tensor(all_token_ids, dtype=torch.long), torch.tensor(all_segments, dtype=torch.long), torch.tensor(self.valid_lens)[idx]]
        else:
            all_token_ids, all_segments = [], []
            for i in idx:
                all_token_ids.append(
                    self.all_token_ids[i] + [self.vocab['<pad>']] * (self.max_len - len(self.all_token_ids[i])))
                all_segments.append(self.all_segments[i] + [0] * (self.max_len - len(self.all_segments[i])))
            return [torch.tensor(all_token_ids, dtype=torch.long), torch.tensor(all_segments, dtype=torch.long), torch.tensor(self.valid_lens)[idx]]

    # 对每一个paragraph（由段落中单词组成的列表）进行数字标签化，并且在句子对(每一个paragraph)开头、中间、结尾加上<cls>、<sep>
    def _get_nsp_data_from_paragraph(self, paragraph):

        # 并且在句子对（每一个paragraph）开头、中间、结尾加上<cls>、<sep>
        def get_tokens_and_segments(tokens_a, tokens_b=None):
            tokens = ['<cls>'] + tokens_a + ['<sep>']
            segments = [0] * (len(tokens_a) + 2)
            if tokens_b is not None:
                tokens += tokens_b + ['<sep>']
                segments += [1] * (len(tokens_b) + 1)
            return tokens, segments

        # 句子对阶段，如果句子对长度大于max_len ,则对其中较长的句子的单词进行截断
        def _truncate_pair_of_tokens(tokens_a, tokens_b):
            while len(tokens_a) + len(tokens_b) > self.max_len - 3:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        # 将单词全部转化为数字
        nums = 0
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b = paragraph[i], paragraph[i + 1]
            if len(tokens_a) + len(tokens_b) + 3 > self.max_len:
                _truncate_pair_of_tokens(tokens_a, tokens_b)
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            tokens_id = self.vocab[tokens]
            self.all_token_ids.append(tokens_id)
            self.all_segments.append(segments)
            self.valid_lens.append(len(tokens))
            nums += 1
        return nums


# 将数据框数据转化为bert需要的数据形式，也即对上面的全部函数进行封装
def load_data_data(test_size, max_len, vocab,valid_file=None):
    if not valid_file:
        Class = data_generate(test_size)
        train_paragraphs_and_labels, test_paragraphs_and_labels, lbl = Class.paragraphs()
        # 训练集数据处理
        train_set = data_preprocess(train_paragraphs_and_labels, max_len, vocab)
        # 测试集数据
        test_set = data_preprocess(test_paragraphs_and_labels, max_len, vocab)
        num_train_iter=len(train_set)
        return train_set,test_set,lbl,num_train_iter
    else:
        # 此时数据为验证数据，不需要进行训练、测试数据划分
        Class = data_generate(0,valid_file)
        train_paragraphs_and_labels, test_paragraphs_and_labels, lbl = Class.paragraphs()
        train_set = data_preprocess(train_paragraphs_and_labels, max_len, vocab)
        return train_set


# 数据生成器，按需生成
def data_iter(set, batch_size,seed=None):
    num_examples = len(set)
    indices = list(range(num_examples))
    # 如果传入seed，则每次生成器打散数据保持不变
    if seed == None:
        pass
    else:
        random.seed(seed)
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield set[batch_indices]


if __name__=="__main__":
    pass