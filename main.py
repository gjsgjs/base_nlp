# -*- coding: utf-8 -*-

# This project is for Roberta model.

"""
    请严格按照234说明的文件格式进行保存
"""

import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from load_data import load_data
from transformers import RobertaTokenizer, AdamW
from parameter import parse_args
from tools import calculate, get_batch, correct_data,collect_mult_event,replace_mult_event,get_test_batch
import random
from model import MLP


args = parse_args()  # load parameters


# -------------------------------- GPU设置 --------------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
# -------------------------------- 日志设置 --------------------------------
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'base__fold-' + str(args.fold) + '__' + t + '.txt'
# args.model = args.model + 'base__fold-' + str(args.fold) + '__' + t + '.pth'
# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')
logger = logging.getLogger(__name__)
def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)
# -------------------------------- 设置随机数 --------------------------------
# set seed for random number
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(args.seed)

printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
# tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
# -------------------------------- 加载数据 --------------------------------
printlog('Loading data')
train_data, dev_data, test_data = load_data(args)
train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)
print('Data loaded')
# -------------------------------- 一些对数据集进行处理的步骤 （此步骤大家可忽略） --------------------------------
# --------------------------------
# 因为数据集中有这种情况的多token事件：put Tompsion on，但事件标注只有put on（在句子中的位置为：_14_16）
# 也就是会出现多token事件的token不连续的情况
# 因此这两个函数的目的是为了让事件的token变连续，即把上面的事件标注变为：put Tompsion on（此时位置为：_14_15_16）
train_data=correct_data(train_data)
dev_data=correct_data(dev_data)
test_data=correct_data(test_data)
# 收集所有事件，以及相应的事件--特殊标识符转换表
# event_dict:special--event；reverse_event_dict:event--special
multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add=collect_mult_event(train_data+dev_data+test_data,tokenizer)
# 将特殊标识符添加到分词器中
tokenizer.add_tokens(special_multi_event_token) #516
args.vocab_size = len(tokenizer)                #50265+len(special_multi_event_token)
# 将句子中的事件用特殊token <a_i> 替换掉，即：He has went to the school.--->He <a_3> the school.
train_data = replace_mult_event(train_data,reverse_event_dict)
dev_data = replace_mult_event(dev_data,reverse_event_dict)
test_data = replace_mult_event(test_data,reverse_event_dict)





# ---------- network ----------
net = MLP(args).to(device)
net.load_state_dict(torch.load("outmodel/base__fold-1__2024-11-01 20.pth"))
net.handler(to_add, tokenizer) # 对新添加的词向量进行初始化 <a_n> 对应的词向量为该事件的平均词向量
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # 不进行权重衰减的参数
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)
# 采用交叉熵损失
cross_entropy = nn.CrossEntropyLoss().to(device)

# 记录验证集最好时，测试集的效果，以及相应的epoch
best_hit1, best_hit3, best_hit10, best_hit50 = 0,0,0,0
dev_best_hit1, dev_best_hit3, dev_best_hit10, dev_best_hit50 = 0,0,0,0
best_hit1_epoch, best_hit3_epoch, best_hit10_epoch, best_hit50_epoch= 0,0,0,0
best_epoch = 0

# 打印一些参数信息
printlog('fold: {}'.format(args.fold))
printlog('batch_size:{}'.format(args.batch_size))
printlog('epoch_num: {}'.format(args.num_epoch))
printlog('initial_t_lr: {}'.format(args.t_lr))
printlog('seed: {}'.format(args.seed))
printlog('wd: {}'.format(args.wd))
printlog('len_arg: {}'.format(args.len_arg))
printlog('len_temp: {}'.format(args.len_temp))
printlog('Start training ...')

if args.test:
    epoch = 0
    all_indices = torch.arange(1, test_size).split(args.batch_size) # 不要随机
    Hit1_t, Hit3_t, Hit10_t, Hit50_t = [], [], [], []

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                            desc='Eval {}'.format(epoch))

    net.eval()
    results = []
    for batch_indices in all_indices:
        progress.update(1)
        # get a batch of dev_data
        batch_arg, mask_arg, mask_indices, labels, candiSet = get_test_batch(test_data, args, batch_indices, tokenizer)
        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction = net(batch_arg, mask_arg, mask_indices, length)

        pred = prediction[0][candiSet[0]].tolist()
        pred.sort(reverse=True)
        rank = []
        for i in range(len(candiSet[0])):
            rank.append(pred.index(prediction[0][candiSet[0][i]]))

        candi_result = [event_dict[k] for k in test_data[int(batch_indices)]['candiSet']]
        batch_result = dict(zip(candi_result, rank))
        results.append(batch_result)

    progress.close()
    data_path = "result/data_epoch{}.json"
    with open(data_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

# 所有数据的候选集

##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()
    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0
    all_loss = 0.0

    Hit1, Hit3, Hit10, Hit50 = [], [], [], []

    all_Hit1, all_Hit3, all_Hit10, all_Hit50 = [], [], [], []

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    #net.eval()
    net.train()
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data) // args.batch_size + 1
    step = 0
    for ii, batch_indices in enumerate(all_indices, 1):
        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, mask_indices, labels, candiSet = get_batch(train_data, args, batch_indices, tokenizer)
        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction = net(batch_arg, mask_arg, mask_indices, length)
        # answer_space：[23702,50265]
        label = torch.LongTensor(labels).to(device)
        # loss
        loss = cross_entropy(prediction,label)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        loss_epoch += loss.item()
        hit1, hit3, hit10, hit50 = calculate(prediction, candiSet, labels, length)
        Hit1 += hit1
        Hit3 += hit3
        Hit10 += hit10
        Hit50 += hit50

        all_Hit1 += hit1
        all_Hit3 += hit3
        all_Hit10 += hit10
        all_Hit50 += hit50
        all_loss += loss.item()
        # import pdb;pdb.set_trace()
        if ii % (100 // args.batch_size) == 0:
            printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit50={:.4f}'.format(
                loss_epoch / (100 // args.batch_size),
                sum(Hit1) / len(Hit1),
                sum(Hit3) / len(Hit3),
                sum(Hit10) / len(Hit10),
                sum(Hit50) / len(Hit50)))
            loss_epoch = 0.0
            Hit1, Hit3, Hit10, Hit50 = [], [], [], []
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    progress.close()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(dev_size).split(args.batch_size)
    Hit1_d, Hit3_d, Hit10_d, Hit50_d = [], [], [], []
    dev_loss = 0.0

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, mask_indices, labels, candiSet = get_batch(dev_data, args, batch_indices, tokenizer)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction = net(batch_arg, mask_arg, mask_indices, length)
        ##
        label = torch.LongTensor(labels).to(device)
        loss = cross_entropy(prediction, label)
        dev_loss += loss.item()
        ##
        hit1, hit3, hit10, hit50 = calculate(prediction, candiSet, labels, length)
        Hit1_d += hit1
        Hit3_d += hit3
        Hit10_d += hit10
        Hit50_d += hit50

    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    # ------------------------------------------------------
    # -------------- 这里不要随机！！！！！！！！！ --------------
    # -----由于测试集的label没有给出，因此运行到253行时会报错-------
    # ---你们需要在验证集最优时，保存好该epoch的测试集的预测结果-----
    # ----因此这一部分的代码需要做调整：保存每个epoch的测试集的预测结果-----
    # ----然后将验证集最优的那个epoch，测试集的预测结果文件提交即可-----
    # ----保存的内容为每条数据候选集事件的预测排名，保存形式见data.json----
    # ------------------------------------------------------
    # all_indices = torch.arange(1, test_size).split(args.batch_size) # 不要随机
    # Hit1_t, Hit3_t, Hit10_t, Hit50_t = [], [], [], []

    # progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
    #                      desc='Eval {}'.format(epoch))

    # net.eval()
    # for batch_indices in all_indices:
    #     progress.update(1)

    #     # get a batch of dev_data
    #     batch_arg, mask_arg, mask_indices, labels, candiSet = get_batch(test_data, args, batch_indices, tokenizer)

    #     batch_arg = batch_arg.to(device)
    #     mask_arg = mask_arg.to(device)
    #     mask_indices = mask_indices.to(device)
    #     length = len(batch_indices)
    #     # fed data into network
    #     prediction = net(batch_arg, mask_arg, mask_indices, length)

    #     hit1, hit3, hit10, hit50 = calculate(prediction, candiSet, labels, length)
    #     Hit1_t += hit1
    #     Hit3_t += hit3
    #     Hit10_t += hit10
    #     Hit50_t += hit50

    # progress.close()

    ############################################################################
    ##################################  result  ##################################
    ############################################################################
    ######### Train Results Print #########
    printlog('-------------------')
    printlog("TIME: {}".format(time.time() - start))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit50={:.4f}'.format(
        # loss_epoch / (100 // args.batch_size),
        all_loss / len(train_data),
        sum(all_Hit1) / len(all_Hit1),
        sum(all_Hit3) / len(all_Hit3),
        sum(all_Hit10) / len(all_Hit10),
        sum(all_Hit50) / len(all_Hit50)))

    ######### Dev Results Print #########
    printlog("DEV:")
    printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit50={:.4f}'.format(
        # loss_epoch / (100 // args.batch_size),
        dev_loss / len(dev_data),
        sum(Hit1_d) / len(Hit1_d),
        sum(Hit3_d) / len(Hit3_d),
        sum(Hit10_d) / len(Hit10_d),
        sum(Hit50_d) / len(Hit50_d)))

    ######### Test Results Print #########
    # printlog("TEST:")
    # printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit50={:.4f}'.format(
    #     loss_epoch / (100 // args.batch_size),
    #     sum(Hit1_t) / len(Hit1_t),
    #     sum(Hit3_t) / len(Hit3_t),
    #     sum(Hit10_t) / len(Hit10_t),
    #     sum(Hit50_t) / len(Hit50_t)))

    # record the best result
    # if sum(Hit1_d) / len(Hit1_d) > dev_best_hit1:
    #     dev_best_hit1 = sum(Hit1_d) / len(Hit1_d)
    #     best_hit1 = sum(Hit1_t) / len(Hit1_t)
    #     best_hit1_epoch = epoch
    #     np.save('predt_hit1.npy', Hit1_t)
    # if sum(Hit3_d) / len(Hit3_d) > dev_best_hit3:
    #     dev_best_hit3 = sum(Hit3_d) / len(Hit3_d)
    #     best_hit3 = sum(Hit3_t) / len(Hit3_t)
    #     best_hit3_epoch = epoch
    #     np.save('predt_hit3.npy', Hit3_t)
    # if sum(Hit10_d) / len(Hit10_d) > dev_best_hit10:
    #     dev_best_hit10 = sum(Hit10_d) / len(Hit10_d)
    #     best_hit10 = sum(Hit10_t) / len(Hit10_t)
    #     best_hit10_epoch = epoch
    #     np.save('predt_hit10.npy', Hit10_t)
    # if sum(Hit50_d) / len(Hit50_d) > dev_best_hit50:
    #     dev_best_hit50 = sum(Hit50_d) / len(Hit50_d)
    #     best_hit50 = sum(Hit50_t) / len(Hit50_t)
    #     best_hit50_epoch = epoch
    #     np.save('predt_hit50.npy', Hit50_t)

    # record the best result
    # 以验证机为准
    if sum(Hit1_d) / len(Hit1_d) > dev_best_hit1:
        dev_best_hit1 = sum(Hit1_d) / len(Hit1_d)
        best_hit1_epoch = epoch
        best_hit1 = dev_best_hit1
    if sum(Hit3_d) / len(Hit3_d) > dev_best_hit3:
        dev_best_hit3 = sum(Hit3_d) / len(Hit3_d)
        best_hit3_epoch = epoch
        best_hit3 = dev_best_hit3
    if sum(Hit10_d) / len(Hit10_d) > dev_best_hit10:
        dev_best_hit10 = sum(Hit10_d) / len(Hit10_d)
        best_hit10_epoch = epoch
        best_hit10 = dev_best_hit10
    if sum(Hit50_d) / len(Hit50_d) > dev_best_hit50:
        dev_best_hit50 = sum(Hit50_d) / len(Hit50_d)
        best_hit50_epoch = epoch
        best_hit50 = dev_best_hit50

    printlog('=' * 20)
    printlog('Best result at hit1 epoch: {}'.format(best_hit1_epoch))
    printlog('Best result at hit3 epoch: {}'.format(best_hit3_epoch))
    printlog('Best result at hit10 epoch: {}'.format(best_hit10_epoch))
    printlog('Best result at hit50 epoch: {}'.format(best_hit50_epoch))
    printlog('Eval hit1: {}'.format(best_hit1))
    printlog('Eval hit3: {}'.format(best_hit3))
    printlog('Eval hit10: {}'.format(best_hit10))
    printlog('Eval hit50: {}'.format(best_hit50))

    # save model
    # 训练循环结束后保存模型参数
    # t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    # args.model = args.model + 'base__fold-' + str(args.fold) + '__' + t + '.pth'
    
    # printlog('Training finished. Saving model...')
    # torch.save(net.state_dict(), args.model)
    # printlog('Model saved to {}'.format(args.model))





# torch.save(state, args.model)

