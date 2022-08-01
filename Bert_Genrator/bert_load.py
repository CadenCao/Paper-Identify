# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/23
# 这个脚本记录了预训练bert模型和经过微调和下游认为魔改的bert模型
import json
from d2l import torch as d2l
import torch
import os
from torch import nn

# 预训练bert模型,下游仅仅加了一个全连接层
def load_pretrained_model():
    data_dir_vocab = r"../bert.small.torch"
    # 模型参数文件链接
    data_dir_predict_params=os.path.join(r"../bert.small.torch",'pretrained.params')
    # 载入原始词汇表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir_vocab,
    'vocab3.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
    vocab.idx_to_token)}
    # 加载bert模型
    bert = d2l.BERTModel(len(vocab), num_hiddens=256, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=512,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=512, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 模型的载入，如果模型的有网络层弃之不用，需要添加参数strict=False
    # (https://blog.csdn.net/weixin_44966641/article/details/120083303)
    bert.load_state_dict(torch.load(data_dir_predict_params))
    return bert, vocab

# 经过微调后的bert模型,也即包括了下游任务
# BERTClassifier仍然是预训练模型
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 12)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
def load_finetune_model(model_file='mlp.params_ce'):
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(r"../bert.small.torch",'vocab3.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens=256, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=512,
                         num_heads=4, num_layers=2, dropout=0,
                         max_len=512, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # net为bert模型框架
    net = BERTClassifier(bert)
    # 载入经过微调后+下游任务的bert模型
    checkpoint = torch.load(os.path.join(r"../Bert_Genrator", model_file))
    net.load_state_dict(checkpoint)
    return net,vocab,model_file

# 下游任务不再是仅仅的一个线性层，而是加了一层双向LSTM的一层双向GRU模型，其他同上面一样
class BERTClassifier_GRU(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier_GRU, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.BiLstm = nn.LSTM(256, 128, 1, dropout=0, bidirectional=True)
        self.BiGru = nn.GRU(256, 256, 1, dropout=0, bidirectional=True)
        self.output = nn.Linear(256, 12)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        encoded_X = encoded_X.permute(1, 0, 2)
        # print(encoded_X.shape)
        BiLstm = self.BiLstm(encoded_X)[0]
        # print(BiLstm.shape)
        BiGru = self.BiGru(BiLstm)[1][-1]
        return self.output(BiGru)
def load_finetune_model_gru(model_file='mlp.params_lstm+gru'):
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(r"../bert.small.torch",'vocab3.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens=256, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=512,
                         num_heads=4, num_layers=2, dropout=0,
                         max_len=512, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    net = BERTClassifier_GRU(bert)
    checkpoint = torch.load(os.path.join(r"../Bert_Genrator", model_file))
    net.load_state_dict(checkpoint)
    return net,vocab,model_file

# 下游是去掉了bert的隐藏层，而增加了一个mean-max-pool层
class BERTClassifier_mmp(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier_mmp, self).__init__()
        self.encoder = bert.encoder
        self.output = nn.Linear(256*2, 12)
    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        # encoded_X为最后一层transformers的全部步长（max_len）的输出，维度为【batch_size,max_len,num_hidden】
        # num_hidden也即单个单词的embedding
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        # 在【max_len,num_hidden】的max_len维度分别求平局和最大值，在级联.avg_pooled维度为【batch_size，num_hidden】
        # max_pooled为【batch_size，num_hidden】
        avg_pooled = encoded_X.mean(1)
        max_pooled = torch.max(encoded_X, dim=1)
        # pooled的维度为【batch_size,2*num_hidden】
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        return self.output(pooled)
def load_finetune_model_mmp():
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(r"../bert.small.torch",
                                                     'vocab3.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens=256, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=512,
                         num_heads=4, num_layers=2, dropout=0.1,
                         max_len=512, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    net = BERTClassifier_mmp(bert)
    checkpoint = torch.load(os.path.join(r"../Bert_Genrator", 'mlp.params_mmp_base'))
    net.load_state_dict(checkpoint)
    return net,vocab

if __name__=="__main__":
    pass




