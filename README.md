# [科大讯飞论文识别算法比赛](https://challenge.xfyun.cn/topic/info?type=abstract&option=cstd)（2022.7）
## 这里使用了亚马逊提供的bert_small预训练模型、scibert预训练模型、传统机器学习模型（主要是TF-IDF）进行集成预测。
### 共包括一下几个文件:
 [***bert.base.torch***](http://d2l-data.s3-accelerate.amazonaws.com/bert.base.torch.zip)：  
* 该文件是亚马逊提供的bert_base模型，12层transformers、mutil_head为12，隐藏层维度为768，包括两个文件,文件pretrained.params是预训练模型参数，3060、6g使用该模型时每次batch_size只能设置为1.使用前4个transformers、8个多头，每个batch_size也只能最高设为20，导致训练时常过长。  

[***bert.small.torch***](http://d2l-data.s3-accelerate.amazonaws.com/bert.small.torch.zip)：  
* 该文件是亚马逊提供的bert_small模型，2层transformers、mutil_head为4，隐藏层维度为512，同样包括两个文件,文件pretrained.params是预训练模型参数，每个batch_size也只能最高可达300，如果GPU不行的话可以尝试该模型。  
	
***Bert_Genrator***：  
* 该文件包括原始数据的处理（data_preprocess.py），使用bert_small模型进行微调、下游任务模型的测试（包括mean-max-pool、bilstm+ligru等等）
（bert_fine_tune.py和bert_load.py）以及微调后的模型参数（mlp.params_ce、mlp.params_fl）

***Bert_Predict***：  
* 该文件包括待预测数据的处理（data_predict_loder.py）、数据的预测（Bert_predict_main.py和TF-IDF.py）、
预测结果文件（result_predict.csv）和验证数据集（data_random_get.csv）

***Bert_Pretrain***：
* 该文件包括bert模型在相关领域的进一步预训练（Pretrain_task.py）和训练参数（pretrained_plus.params）
	
***data***:  
* 该文件是科大讯飞提供的数据，包括test(待预测并提交的数据)，train（全部训练数据）、train1、train2则是本人训练过程中使用到的数据，提交实例文件则是科大讯飞规定需要提交的数据格式。

[***scibert***](https://github.com/huggingface/transformers):  
* 该文件是使用其他的项目相关领域的bert预训练模型（scibert），训练效果同Bert_Genrator相近。***Bert_Genrator、Bert_Pretrain代码都有相应的解释，由于scibert文件和Bert_Pretrain同前两个文件代码雷同过多，因此没有过多去赘述，可参考前两个文件的代码。***

###  **由于github上传文件大小由限制，因此将数据文件以及模型参数文件均已删除**



