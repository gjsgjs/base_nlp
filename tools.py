import torch
# from numpy import random


def getTemplate(args, data):
    edge = data['edge'][:-1] if len(data['edge'])<=(args.len_arg)//10 else data['edge'][0:(args.len_arg)//10]
    causeRatio = args.cause_ratio/(args.cause_ratio+args.becausedby_ratio)
    causeRel = edge[0:int(len(edge)*causeRatio)]
    becausedbyRel = [[rel[-1], 'be casued by', rel[0]] for rel in edge[int(len(edge)*causeRatio):]]
    template = ''
    relation = [] + causeRel + becausedbyRel
    for rel in relation:
        eId1 = rel[0]
        eId2 = rel[-1]
        rl = data['node'][eId1][5] + ' ' + rel[1] + ' ' + data['node'][eId2][5]
        template = template + rl + ' , '
    maskRel = data['edge'][-1]
    return template + data['node'][maskRel[0]][5] + ' ' + maskRel[1] + ' <mask> .', relation + [maskRel]

# tokenize sentence and get event idx
def get_batch(data, args, indices, tokenizer):
    batch_idx = []
    batch_mask = []
    mask_indices = []   # mask所在位置
    labels = []         # 存储真实事件标签的id
    candiSet = []  # 存储候选节点的id
    for idx in indices:
        candi = [tokenizer.encode(data[idx]['candiSet'][i])[1] for i in range(len(data[idx]['candiSet']))]
        template, relation = getTemplate(args, data[idx])

        encode_dict = tokenizer.encode_plus(
            template,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        label = tokenizer.encode(data[idx]['candiSet'][data[idx]['label']])[1]
        labels.append(label)
        candiSet.append(candi)

        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    return batch_idx, batch_mask, mask_indices, labels, candiSet

#### 
# tokenize sentence and get event idx
def get_test_batch(data, args, indices, tokenizer):
    batch_idx = []
    batch_mask = []
    mask_indices = []   # mask所在位置
    labels = []         # 存储真实事件标签的id
    candiSet = []  # 存储候选节点的id
    for idx in indices:
        candi = [tokenizer.encode(data[idx]['candiSet'][i])[1] for i in range(len(data[idx]['candiSet']))]
        template, relation = getTemplate(args, data[idx])

        encode_dict = tokenizer.encode_plus(
            template,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        # import pdb; pdb.set_trace()
        # label = tokenizer.encode(data[idx]['candiSet'][data[idx]['label']])[1]
        # labels.append(label)
        candiSet.append(candi)

        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    return batch_idx, batch_mask, mask_indices, None, candiSet


# calculate Hit@1 Hit@3 Hit@10
def calculate(prediction, candiSet, labels, batch_size):
    hit1, hit3, hit10, hit50 = [], [], [], []
    for i in range(batch_size):
        predtCandi = prediction[i][candiSet[i]].tolist()
        label = candiSet[i].index(labels[i])
        labelScore = predtCandi[label]
        predtCandi.sort(reverse=True)
        rank = predtCandi.index(labelScore)
        hit1.append(int(rank<1))
        hit3.append(int(rank<3))
        hit10.append(int(rank<10))
        hit50.append(int(rank < 50))

    return hit1, hit3, hit10, hit50


def isContinue(id_list):
    for i in range(len(id_list) - 1):
        if int(id_list[i]) != int(id_list[i + 1]) - 1:
            return False
    return True


def doCorrect(data):
    for i in range(len(data)):
        eId = data[i][8].split('_')[1:]
        if not isContinue(eId):
            s_1 = data[i][6].split()
            event1 = s_1[int(eId[0]):int(eId[-1]) + 1]
            event1 = ' '.join(event1)
            event1 += ' '  # 在这里加空格是 因为其它event后面都有一个空格，这里仅仅是为了与它们保持一致
            new_e1_id = [str(i) for i in range(int(eId[0]), int(eId[-1]) + 1)]
            temp = ''
            for ii in new_e1_id:
                temp += s_1[int(ii)] + ' '
            assert event1 == temp
            event_place1 = '_' + '_'.join(new_e1_id)
            sentence = (
            data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], event1, data[i][6], data[i][7],
            event_place1)
            data[i] = sentence
    return data

def correct_data(dataSet):
    for i in range(len(dataSet)):
        dataSet[i]['node'] = doCorrect(dataSet[i]['node'])
        # dataSet[i]['candiSet'] = doCorrect(dataSet[i]['candiSet'])
    return dataSet

def doCollect(data, tokenizer, multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict, flag_candi=0):
    for sentence in data:
        if flag_candi == 0:
            event = sentence[5]
        else:
            event = sentence
        # 为了方便后续替换，这里选择把所有的事件都进行替换
        if event not in multi_event:
            multi_event.append(event)
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = multi_event[-1]
            reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
    return multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict


def collect_mult_event(train_data, tokenizer):
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    for sentence in train_data:
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(sentence['node'][:-1],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict)
        multi_event, to_add, special_multi_event_token, event_dict, reverse_event_dict = doCollect(
                                                                                                   sentence['candiSet'][:-1],
                                                                                                   tokenizer,
                                                                                                   multi_event, to_add,
                                                                                                   special_multi_event_token,
                                                                                                   event_dict,
                                                                                                   reverse_event_dict,1)
    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add


def doReplace(data, reverse_event_dict):
    for i in range(len(data)):
        # assert data[i][5] in reverse_event_dict
        if data[i][5] in reverse_event_dict:
            sent = data[i][6].split()
            eId = data[i][8].split('_')[1:]
            eId.reverse()
            for id in eId:
                sent.pop(int(id))
            sent.insert(int(eId[-1]), reverse_event_dict[data[i][5]])
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], reverse_event_dict[data[i][5]], " ".join(sent),
                            data[i][7], '_' + eId[-1])
            data[i]=sentence
    return data

def replace_mult_event(data, reverse_event_dict):
    for i in range(len(data)):
        data[i]['node'] = doReplace(data[i]['node'], reverse_event_dict)
        temp = [reverse_event_dict[e] for e in data[i]['candiSet']]
        data[i]['candiSet'] = temp
    return data






if __name__ == '__main__':
    pass
