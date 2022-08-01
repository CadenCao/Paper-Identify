# [scibert](https://github.com/huggingface/transformers) 预训练模型
该文件是使用了不同的scibert预训练模型，其中***config.json***、***vocab.txt***、***pytorch_model.bin***是原始文件，***mlp.paramsmlp***.***params1***、***mlp.params_ce***是微调后生成的模型参数文件，***sci_result_predict***是预测数据生成的文件（同***bert_predict***文件下的***result_predict***文件）。***scibert_data_preprocess***、***scibert_fine_tune***、***scibert_predict***的作用分别同***Bert_Genrator***文件下的***data_preprocess***、***bert_fine_tune***、***bert_load***.但不同的是对于原始数据数据我这里使用的是***hugging face***框架下面的***transformers***类。  

最终模型预测结果同***Bert_Genrator***差距不大，均在0.9左右。
