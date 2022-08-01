# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/24
# 这个是bert进行微调和训练的脚本

from torch import nn
import torch
from d2l import torch as d2l
from bert_load import load_finetune_model,load_pretrained_model,load_finetune_model_mmp,\
    load_finetune_model_gru
from data_preprocess import load_data_data,data_iter

# 没有经过微调的初始bert模型，直接在bert预训练模型下面加了一个线性层
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        # bert预训练模型的输出结果
        self.encoder = bert.encoder
        # bert预训练模型中的最后一层隐藏层
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 12)
    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

# 没有经过微调的初始bert模型，在bert预训练模型的基础上，下游任务直接加了一个双向LSTM和双向GRU模型，
# 最后一层再接了一个线性层
class BERTClassifier_GRU(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier_GRU, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.BiLstm=nn.LSTM(256,128,1,dropout=0,bidirectional=True)
        self.BiGru=nn.GRU(256,256,1,dropout=0,bidirectional=True)
        self.output = nn.Linear(256, 12)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        encoded_X=encoded_X.permute(1,0,2)
        # print(encoded_X.shape)
        BiLstm=self.BiLstm(encoded_X)[0]
        # print(BiLstm.shape)
        BiGru=self.BiGru(BiLstm)[1][-1]
        return self.output(BiGru)
#
# 在bert预训练下游任务中添加mean-max-pool层
# class BERTClassifier_mmp(nn.Module):
#     def __init__(self, bert):
#         super(BERTClassifier_mmp, self).__init__()
#         self.encoder = bert.encoder
#         self.output = nn.Linear(256*2, 12)
#     def forward(self, inputs):
#         tokens_X, segments_X, valid_lens_x = inputs
#         encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
#         avg_pooled = encoded_X.mean(1)
#         max_pooled = torch.max(encoded_X, dim=1)
#         pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
#         return self.output(pooled)

# 学习率随epoch变换函数-普通动态学习率
# class SquareRootScheduler:
#     def __init__(self, lr=0.1):
#         self.lr = lr
#     def __call__(self, num_update):
#         return self.lr * pow((num_update + 1), -2)

# 学习率随epoch变换函数-余弦调度因子
# class CosineScheduler:
#     def __init__(self, max_update, base_lr=0.01, final_lr=0,warmup_steps=0, warmup_begin_lr=0):
#         self.base_lr_orig = base_lr
#         self.max_update = max_update
#         self.final_lr = final_lr
#         self.warmup_steps = warmup_steps
#         self.warmup_begin_lr = warmup_begin_lr
#         self.max_steps = self.max_update - self.warmup_steps
#     def get_warmup_lr(self, epoch):
#         increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
#         return self.warmup_begin_lr + increase
#     def __call__(self, epoch):
#         if epoch < self.warmup_steps:
#             return self.get_warmup_lr(epoch)
#         if epoch <= self.max_update:
#             self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
#                 (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
#             return self.base_lr

# 学习率随epoch变换函数-多因子调度器
class FactorScheduler:
    def __init__(self, factor=1.0, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr
    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr



# focal_loss 损失函数，为了防止计算出来的概率分布过于偏向与中间部分（0.5），本文添加了惩罚措施，
# 即在focal loss函数分母添加了一个惩罚措施，即((Max/Second_Max)**self.fixed)，self.fixed是一个超参数，
# 其中max是计算出的各个样本标签概率中最大的概率值，Second_Max为第二大的概率值，因为样本的真实标签一般就分布在
# 最大或第二大其中之一
class focal_loss():
    def __init__(self,gammer,fixed=0.):
        self.gammer = gammer
        self.fixed=fixed
    def __call__(self,y_hat,y, *args, **kwargs):
        y_hat_softmax = nn.functional.softmax(y_hat,dim=1)
        if self.fixed:
            Sort=y_hat_softmax.sort(1,True)
            Max=Sort[0][:,0]
            Second_Max=Sort[0][:,1]
        y_true_prob = y_hat_softmax[range(len(y_hat)),y]
        def cross_loss():
            return -(1-y_true_prob)**self.gammer*torch.log(y_true_prob) if not self.fixed else \
                -(1 - y_true_prob) ** self.gammer * torch.log(y_true_prob)/((Max/Second_Max)**self.fixed)
        return cross_loss()

# 正式开始训练模型
def train_bert(train_set, test_set,batch_size,net,loss,num_epochs,num_batches,valid_set=None,devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    print("-"*100 + f"训练样本数目：{len(train_set)},验证样本数目：{len(valid_set)}")
    # 设置的初始学习率，如果是使用的预训练模型，该值可以大一些，可以从0.01开始，到1e-6,如果是使用的经过微调后的预训练模型，
    # 该值可以从0.0001开始到1e-6
    lr_start = 0.000001
    scheduler=FactorScheduler(factor=0.8, stop_factor_lr=1e-8, base_lr=lr_start)
    # 针对bert模型不同网络层，设置不同的学习率，其他层由于是经过了预训练，所以统一，而最后一层线性层由于是初始化，所以将学习率设的大一些
    # 最后一层output的参数
    output_params = list(map(id, net.module.output.parameters()))
    # 非最后一层参数
    base_params = filter(lambda p: id(p) not in output_params, net.module.parameters())
    trainer = torch.optim.SGD(
        # 其他层学习率全部统一
        [{'params': base_params, "lr": lr_start},
         # 最后一层线性层学习率较大一些
         {'params': net.module.output.parameters(), 'lr': lr_start}])
    seed=None
    for epoch in range(num_epochs):
        # 对于数据生产，没有使用Dataloader类，因为使用该类是一次性先全部生成了讲过padding后的数据，数据量很大，可能加载很慢，而且必须
        # 放在__name__ == "__main__"下面才不会报错，因此综合考虑，本文这里使用了data_iter函数，该函数定义了一个生成器，每次需要batchsize
        # 数量的数据，就生成该部分经过padding后的数据，这里我将train_iter放在了epoch现在下面，这意味这每个epoch都会生成不同的样本顺序，但是
        # 样本的仍然保存不变，具体可以看data_preprocess文件关于数据处理部分
        train_iter = data_iter(train_set, batch_size,seed)
        # 第一个epoch的学习率保持为初始学习率
        lr = scheduler(epoch) if epoch!=0 else lr_start
        # 不同bert网络层的学习率设置
        trainer.param_groups[0]["lr"] = lr
        trainer.param_groups[1]["lr"] = 1.25*lr
        # Accumulator该函数用来记录模型中变量的累加变化情况，如损失、精确率等等
        metric = d2l.Accumulator(4)
        # features为数量为【batchsize,max_len,num_hiddens】维度的数据，其中batchsize为批量大小，我这里一般设置为100，过大容易导致过拟合
        # 过小导致收敛过慢。max_len为一个句子对的最大长度，预训练模型为512，num_hiddens，为隐藏层或者词嵌入维度（embedding_size）
        for i, (features, labels) in enumerate(train_iter):
            # 单个batch_size的训练
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            # 将需要查看的参数添加至metric
            metric.add(l, acc, labels.shape[0], labels.numel())
            # 每隔10个batch_size或最后一个样本就进行打印
            if (i + 1) % (10) == 0 or i == num_batches - 1:
                # 打印出epoch的序号以及bert网络层学习率
                print("-" * 30 + f" epoch:{epoch},Lr_rate:{lr:.10f},last_lr_rate:{trainer.param_groups[1]['lr']:.10f}")
                # 打印出batch_size序号
                print("-" * 10 + f" batch_size:{i}")
                # 每隔10个batch_size打印当前epoch下的损失率，训练精度
                print(f'loss:{metric[0] / metric[2]:.3f}, train acc:'f'{metric[1] / metric[3]:.3f}','\n\n')
        # 这里计算了测试样本（这里我命名为valid_set）的准确，这个测试样本是永远固定的，并且不参与模型训练，容易模拟真实模型的准确率
        if valid_set:
            # 过程同上
            valid_iter=data_iter(valid_set,batch_size,seed)
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            print("-" * 50+f'loss:{metric[0] / metric[2]:.3f}, train acc:'f'{metric[1] / metric[3]:.3f}, valid acc:{valid_acc:.4f}')

# 单个batch_size的训练
def train_batch_ch13(net, X, y, loss, trainer, devices):
    # 将样本迁移到GPU上
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    # y是一个一维张量，每个数值为对应样本的标签
    y = y.to(devices[0])
    # 样本训练
    net.train()
    # 梯度先归零
    trainer.zero_grad()
    # 对全部样本进行预测，返回一个大小为【batch_size，num_label】维度的数据，其中num_label为每个样本中每个标签概率的预测
    pred = net(X)
    # 计算损失值
    l = loss(pred, y)
    # 进行反向传播
    l.sum().backward()
    trainer.step()
    # 加总全部的损失
    train_loss_sum = l.sum()
    # 计算在该batch_size下面的准确率
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


if __name__ == "__main__":
    # 有GPU就用GPU
    devices = d2l.try_all_gpus()

    # 导入预训练模型（如果是初次预训练，直接导入这个模型）
    # bert, vocab = load_pretrained_model()
    # net = BERTClassifier(bert)

    # 这个是经过微调后的Bert模型，如果是需要进一步对微调后的模型进一步训练或者对数据的预测。可以导入该模型，
    # 具体模型的选择可以参见bert_load文件
    net,vocab,_=load_finetune_model()
    # max_len为每个句子对的最长长度，test_size为将训练样本分割为测试集和验证集
    batch_size, max_len, test_size = 90, 512, 0.02
    train_set, test_set,lbl,num_batches = load_data_data(test_size, max_len, vocab)
    # 生成测试集数据，全部数据用以验证
    valid_set = load_data_data(0, max_len, vocab,valid_file="../Bert_Predict/data_random_get.csv")

    # epoch数量
    num_epochs=10
    # 损失函数的选择，交叉熵和focal loss
    loss=nn.CrossEntropyLoss(reduction='none')
    # loss=focal_loss(2,fixed=0.5)

    # 开始训练
    train_bert(train_set, test_set,batch_size,net,loss,num_epochs,num_batches//batch_size+1,valid_set=valid_set,devices=devices)
    # 模型的保存
    torch.save(net.state_dict(), 'mlp.params_ce')