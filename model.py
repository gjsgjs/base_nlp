# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
import numpy as np
from transformers import GPT2LMHeadModel
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


# class MLP(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
#         self.roberta_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
#         # 在保留原始词嵌入的基础上，增加新的词汇并为它们初始化词嵌入
#         self.roberta_model.resize_token_embeddings(args.vocab_size)
#         for param in self.roberta_model.parameters():
#             param.requires_grad = True

#         self.hidden_size = 768

#         self.vocab_size = args.vocab_size

#     # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
#     def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
#         word_emb = self.roberta_model.roberta.embeddings.word_embeddings(batch_arg).to(device)
#         temp_emb = self.roberta_model(attention_mask = arg_mask, inputs_embeds = word_emb)[0].to(device)

#         prediction = torch.tensor([]).to(device)
#         for i in range(batch_size):
#             e_emb = self.extract_event(temp_emb[i], mask_indices[i])
#             if i == 0:
#                 prediction = e_emb
#             else:
#                 prediction = torch.cat((prediction, e_emb),dim=0)
#         return prediction


#     def extract_event(self, embed, mask_idx):
#         mask_embed = embed[mask_idx]
#         mask_embed = torch.unsqueeze(mask_embed, 0)
#         return mask_embed
#     # 多token事件特殊标识符采用平均初始化
#     def handler(self, to_add, tokenizer):
#         # da shape: [53783, 768] 53783是原始词汇表大小，768是词嵌入维度
#         da = self.roberta_model.roberta.embeddings.word_embeddings.weight
#         for i in to_add.keys():
#             l = to_add[i]
#             with torch.no_grad():
#                 temp = torch.zeros(self.hidden_size).to(device)
#                 for j in l:
#                     temp += da[j]
#                 temp /= len(l)

#                 da[tokenizer.convert_tokens_to_ids(i)] = temp
        # example
        # i = '<a_0>' , to_add[i] = [5234, 9399], '<a_0>' <--> prevent
        # tokenizer.convert_ids_to_tokens([5234,9399]) = ['pre', 'vent']
        # temp = (da[5234] + da[9399]) / 2
        # da[tokenizer.convert_tokens_to_ids('<a_0>')] = temp




class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        # 在保留原始词嵌入的基础上，增加新的词汇并为它们初始化词嵌入
        self.gpt2_model.resize_token_embeddings(args.vocab_size)
        for param in self.gpt2_model.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size

    # batch_arg:句子分词id，arg_mask:句子分词掩码，mask_indices:[MASK]在分词id中的位置，event_group:事件id集合
    def forward(self, batch_arg,mask_indices, batch_size):
        # import ipdb;ipdb.set_trace()
        # 将输入转换为 tensor
        input_ids = torch.tensor(batch_arg).unsqueeze(0)  # 增加一个维度以符合输入格式
        # 将输入传入模型，获取输出
        
        outputs = self.gpt2_model(input_ids)
        logits = outputs.logits[0]

        # 获取预测结果
        temp_emb = logits
        # prediction = torch.tensor([]).to(device)

        for i in range(batch_size):
            e_emb = self.extract_event(temp_emb[i], mask_indices[i])
            if i == 0:
                prediction = e_emb
            else:
                prediction = torch.cat((prediction, e_emb),dim=0)
        return prediction
       


    def extract_event(self, embed, mask_idx):
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed
    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        # da shape: [50257, 768] 50257是GPT-2的词汇表大小，768是词嵌入维度
        da = self.gpt2_model.transformer.wte.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp