# 一*-coding : utf-8 -*-
# author: Canden Cao time: 2022/7/30
import torch
import transformers
from scibert_data_preprocess import load_data_data,data_iter
from torch import nn
from d2l import torch as d2l
import os

def load_pretrained_model():
    model_config = transformers.BertConfig.from_pretrained(r"./config.json")
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    bert = transformers.BertModel.from_pretrained(r"./pytorch_model.bin", config=model_config)
    for param in bert.parameters():
        param.requires_grad_(True)
    return bert

def sci_load_finetune_model(model_file='mlp.params_ce'):
    bert = load_pretrained_model()
    net = Model(bert)
    checkpoint = torch.load(os.path.join(r"./", model_file))
    net.load_state_dict(checkpoint)
    return net

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

class FactorScheduler:
    def __init__(self, factor=1.0, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr
    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr
def tokenizer_generate(set):
    return tokenizer.batch_encode_plus(batch_text_or_text_pairs=set,
                                                    truncation_strategy="longest_first",
                                                    max_length=512,
                                                    padding='max_length',
                                                    return_tensors="pt")
def train_bert(train_set,batch_size,net,loss,num_epochs,num_batches,valid_set=None,devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    print("-"*100 + f"训练样本数目：{len(train_set[0])},验证样本数目：{len(valid_set[0])}")
    lr_start = 0.0001
    scheduler=FactorScheduler(factor=0.7, stop_factor_lr=1e-8, base_lr=lr_start)
    output_params = list(map(id, net.module.output.parameters()))
    base_params = filter(lambda p: id(p) not in output_params, net.module.parameters())
    trainer = torch.optim.SGD(
        [{'params': base_params, "lr": lr_start},
         {'params': net.module.output.parameters(), 'lr': lr_start}])
    for epoch in range(num_epochs):
        train_iter = data_iter(train_set, batch_size)
        lr = scheduler(epoch) if epoch!=0 else lr_start
        trainer.param_groups[0]["lr"] = lr
        trainer.param_groups[1]["lr"] = 1.25*lr
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            tokenizer_features=tokenizer_generate(features)
            l, acc = train_batch_ch13(net, tokenizer_features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (10) == 0 or i == num_batches - 1:
                print("-" * 30 + f" epoch:{epoch},Lr_rate:{lr:.10f},last_lr_rate:{trainer.param_groups[1]['lr']:.10f}")
                print("-" * 10 + f" batch_size:{i}")
                print(f'loss:{metric[0] / metric[2]:.3f}, train acc:'f'{metric[1] / metric[3]:.3f}','\n\n')
        if valid_set:
            valid_iter = data_iter(valid_set, batch_size)
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
            print("-" * 100 + f'valid acc:{valid_acc:.4f}')


def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X=tokenizer_generate(X)
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


if __name__ == "__main__":

    devices = d2l.try_all_gpus()

    net = sci_load_finetune_model()

    batch_size, max_len, test_size = 20, 512, 0.02
    x_train, x_test, y_train, y_test, lbl, num_batches = load_data_data(test_size)
    x_valid,y_valid = load_data_data(0, valid_file="../Bert_Predict/data_random_get.csv")
    tokenizer = transformers.BertTokenizer.from_pretrained(r"./vocab.txt")

    train_set,valid_set=(x_train,y_train),(x_valid,y_valid)
    # 微调参数设置
    num_epochs=15
    loss=nn.CrossEntropyLoss(reduction='none')
    # loss=focal_loss(2,fixed=0.5)

    # 训练
    train_bert(train_set,batch_size,net,loss,num_epochs,num_batches//batch_size+1,valid_set=valid_set,devices=devices)

    torch.save(net.state_dict(), 'mlp.params_ce_translate')



