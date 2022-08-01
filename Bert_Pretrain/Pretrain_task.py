# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/28
import os
import d2l.torch
import pandas as pd
import torch
from d2l import torch as d2l
import re
from torch import nn
import json
import random
'''
数据处理

'''
class data_generate():
    def __init__(self,data_file = r'../data/train.csv'):
        self.data_file = data_file
    def data_read(self):
        train_df = pd.read_csv(self.data_file, sep=',')
        train_df['Abstract'] = train_df['Abstract'].fillna('')
        text = []
        for i in range(len(train_df)):
            Title1 = train_df.loc[i, 'Title'].strip().replace('\n', "").replace("[", "").\
                replace("]", "").replace("-"," ").lower()
            abstract = train_df.loc[i, 'Abstract']
            abstract_plus = re.sub(" {2,}|[^a-z,.:;?]{3,}", " ", re.sub(r"\(.*?\)|\[.*?]|[)(\-]", " ",
                                abstract.strip().replace("\n", " ").replace(","," , ").replace(":", " : ").lower()))
            text.append((Title1 + " . " + abstract_plus.replace('!', " . ").replace('?', " . ").replace( "; ", " ; ")))
        return text
    def paragraphs(self):
        train_paragraphs = self.data_read()
        # 将X中每个段落变成句子词表形式
        def exception(line):
            raise Exception(f'the lines is "{line}"')
        train_paragraphs = [line.split('. ') if len(line.split('. ')) >= 2 else exception(line) for line in train_paragraphs ]
        return train_paragraphs


class data_process(torch.utils.data.Dataset):
    def __init__(self, train_paragraphs, max_len, vocab):
        # 将每个段落中的句子字符串以空白符进行分割，自此paragraphs形成3维，[a,b,c]，其中a表示段落个数,b表示段落中句子个数，c表示句子的词元个数
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in train_paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = vocab
        examples=[]
        for paragraph in paragraphs:
            examples.extend(self._get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语⾔模型任务的数据
        self.examples = [(d2l._get_mlm_data_from_tokens(tokens, self.vocab)+ (segments, is_next)) for tokens, segments, is_next in examples]


    def __getitem__(self, idx):
        return self._pad_bert_inputs(self.examples,max_len,idx)

    def __len__(self):
        return len(self.examples)

    def _pad_bert_inputs(self,examples,max_len,idx):
        max_num_mlm_preds=round(max_len*0.15)
        if not isinstance(idx, (list, tuple)):
            all_token_ids = self.all_token_ids[idx] + [self.vocab['<pad>']] * (
                    self.max_len - len(self.all_token_ids[idx]))
            all_segments = self.all_segments[idx] + [0] * (self.max_len - len(self.all_segments[idx]))
            return [torch.tensor(all_token_ids, dtype=torch.long), torch.tensor(all_segments, dtype=torch.long),
                    torch.tensor(self.valid_lens)[idx]]
        else:
            all_token_ids, all_segments,valid_lens,all_pred_positions,all_mlm_weights,all_mlm_labels,nsp_labels = [],[],[],[],[],[],[]
            for (token_ids, pred_positions, mlm_pred_label_ids, segments,is_next) in [examples[i] for i in idx]:
                all_token_ids.append(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)))
                all_segments.append(segments + [0] * (max_len - len(segments)))
                # `valid_lens` excludes count of '<pad>' tokens
                valid_lens.append(len(token_ids))
                all_pred_positions.append(pred_positions +[0] * (max_num_mlm_preds - len(pred_positions)))
                # Predictions of padded tokens will be filtered out in the loss via
                # multiplication of 0 weights
                all_mlm_weights.append([1.0] * len(mlm_pred_label_ids) +[0.0] * (max_num_mlm_preds - len(pred_positions)))
                all_mlm_labels.append(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)))
                nsp_labels.append(is_next)
            return (torch.tensor(all_token_ids,dtype=torch.long),torch.tensor(all_segments, dtype=torch.long),
                    torch.tensor(valid_lens, dtype=torch.float32),torch.tensor(all_pred_positions, dtype=torch.long),
                    torch.tensor(all_mlm_weights,dtype=torch.float32),torch.tensor(all_mlm_labels, dtype=torch.long),
                    torch.tensor(nsp_labels, dtype=torch.long))

    def _get_nsp_data_from_paragraph(self,paragraph, paragraphs, vocab, max_len):

        def _truncate_pair_of_tokens(tokens_a, tokens_b):
            while len(tokens_a) + len(tokens_b) > max_len - 3:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
            return tokens, segments

        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = d2l._get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                _truncate_pair_of_tokens(tokens_a,tokens_b)
            tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))

        return nsp_data_from_paragraph

'''
重新预训练Bert
'''
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,valid_lens_x.reshape(-1),pred_positions_X)
    # 计算遮蔽语⾔模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下⼀句⼦预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
def train_bert(train_set, net, loss, vocab_size, devices, epoch):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    step, timer = 0, d2l.Timer()
    metric = d2l.Accumulator(4)
    lr_strat = 0.001
    for j in range(epoch):
        lr_strat*=0.8
        train_iter = data_iter(train_set, batch_size)
        trainer = torch.optim.Adam(net.parameters(), lr=lr_strat)
        for i,(tokens_X, segments_X, valid_lens_x, pred_positions_X,mlm_weights_X, mlm_Y, nsp_y) in enumerate(train_iter):
            print(i,j,lr_strat)
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            print(f'MLM loss {metric[0] / metric[3]:.3f}, '
                  f'NSP loss {metric[1] / metric[3]:.3f}')
            print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
                  f'{str(devices)}')
def train_datast(max_len, vocab):
    Class = data_generate()
    train_paragraphs = Class.paragraphs()
    train_set = data_process(train_paragraphs, max_len, vocab)
    print(f"train_set长度:{len(train_set)}")
    return train_set
def data_iter(set, batch_size):
    num_examples = len(set)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield set[batch_indices]

if __name__=="__main__":
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(r"../bert.small.torch", 'vocab2.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    net = d2l.BERTModel(len(vocab), num_hiddens=256, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=512,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=512, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    data_dir_vocab = r"../bert.small.torch"
    data_dir_predict_params = os.path.join(r"./bert.small.torch", 'pretrained.params')
    net.load_state_dict(torch.load(data_dir_predict_params))

    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()
    batch_size,max_len,epoch=40,512,1

    train_set=train_datast(max_len,vocab)

    train_bert(train_set, net, loss, len(vocab), devices, epoch)
    torch.save(net.state_dict(), 'pretrained_plus.params1')