#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:09:30 2023

@author: yuanbeiming
"""
# from attr_regress_with_rule import attr_regress_plus_all_model_center as model_vic

import make_generate_all_attribute as make_data

if make_data.dataset == 'distribute_four':

    from attr_regress_with_rule import attr_regress_plus_all_model_d4 as model_vic
    

elif make_data.dataset == 'distribute_nine':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model as model_vic
    
    
elif make_data.dataset == 'center_single':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model_center as model_vic
    

else:
    
    print('dataset error')


import torch

import torch.nn.functional as F

from tqdm import tqdm




dataset = make_data.dataset

model_color = model_vic.VIC_constant(attr = 'color')

model_color.load_state_dict(torch.load('./model_attr_model_plus_color_'+str(len(make_data.train_file))+'_Datasets_'+ dataset +'_color_now.pt', map_location = 'cpu'))

model_type = model_vic.VIC_constant(attr = 'type')

model_type.load_state_dict(torch.load('./model_attr_model_plus_type_'+str(len(make_data.train_file))+'_Datasets_'+ dataset +'_type_now.pt', map_location = 'cpu'))

model_size = model_vic.VIC_constant(attr = 'size')

model_size.load_state_dict(torch.load('./model_attr_model_plus_size_'+str(len(make_data.train_file))+'_Datasets_'+ dataset +'_size_now.pt', map_location = 'cpu'))

model_color.cuda()

model_type.cuda()

model_size.cuda()

model_color.eval()

model_type.eval()

model_size.eval()

num_entity = model_color.low_dim

val_loader, num_val = make_data.raven_loader(1, train = False, val = False)

def choose_accuracy(model, *out, rule_label):
    
    x, x_regression, rule, txt = out
    
    rule = rule[0]
    
    x_embedding = model.Embedding(x).reshape(-1, 9, model.low_dim*model.attribute)
    
    # print(x.shape)
    
    txt = txt.mean(dim = 0, keepdim = True)
    
    x_regression = F.gumbel_softmax(x_regression, dim = -1).argmax(dim = -1)
    
    attr_regression = x_regression
    
    x_regression = model.Embedding(x_regression).reshape(-1, 1, model.low_dim*model.attribute)
    
    x_embedding = torch.cat((x_embedding[:,:8], x_regression), dim = 1)
    
    assert x_embedding.shape[1] == 9
    
    
    rule_1 = model.rule_attr_clip(x_embedding[:,3:])
    
    rule_2 = model.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]])
    
    r = F.cosine_similarity(txt, rule, dim = -1)
    
    r_1 = F.cosine_similarity(txt, rule_1, dim = -1)
    
    r_2 = F.cosine_similarity(txt, rule_2, dim = -1)
    
    r = (r.argmax(dim = -1) == rule_label).float()
    
    r_1 = (r_1.argmax(dim = -1) == rule_label).float()
    
    r_2 = (r_2.argmax(dim = -1) == rule_label).float()
    
    r = r + r_1 + r_2
    
    return (r.eq(3)).float().sum(), attr_regression

def choose_accuracy_with_position(model, *out, rule_label, position):
    
    x, x_regression, rule, txt = out
    
    rule = rule[0]
    
    assert model.attribute == 7
    
    x_embedding = model.Embedding(x).reshape(-1, 9, model.low_dim*model.attribute)
    
    # print(x.shape)
    
    txt = txt.mean(dim = 0, keepdim = True)
    
    # print(x_regression.shape)
    
    x_regression = F.gumbel_softmax(x_regression[:,:,:-1], dim = -1).argmax(dim = -1)#1,9,6 -> 1,9
    
    
    
    # print(x_regression.shape)
    
    x_regression = (x_regression*position + (1-position)*(model.attribute - 1)).long()
    
    # print(x_regression.shape)
    
    attr_regression = x_regression
    
    x_regression = model.Embedding(x_regression).reshape(-1, 1, model.low_dim*model.attribute)
    
    x_embedding = torch.cat((x_embedding[:,:8], x_regression), dim = 1)
    
    assert x_embedding.shape[1] == 9
    
    
    rule_1 = model.rule_attr_clip(x_embedding[:,3:])
    
    rule_2 = model.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]])
    
    r = F.cosine_similarity(txt, rule, dim = -1)
    
    r_1 = F.cosine_similarity(txt, rule_1, dim = -1)
    
    r_2 = F.cosine_similarity(txt, rule_2, dim = -1)
    
    r = (r.argmax(dim = -1) == rule_label).float()
    
    r_1 = (r_1.argmax(dim = -1) == rule_label).float()
    
    r_2 = (r_2.argmax(dim = -1) == rule_label).float()
    
    r = r + r_1 + r_2
    
    return (r.eq(3)).float().sum(), attr_regression





def discrimnater(t,s,c, t_rule, s_rule, c_rule):
    
    MAX_iter = 500
    
    out = model_color(c)
    
    for i in range(MAX_iter):
        
        choose_right, c_regression = choose_accuracy(model_color, *out, rule_label = c_rule)
        
        
        
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            break
        
    if i == MAX_iter - 1:
        return 0
            
            
    pre_position = (c_regression != 10).float()
    assert pre_position.shape[1] == num_entity
#%%
    out = model_type(t)
    
    for i in range(MAX_iter):
        
        choose_right, t_regression = choose_accuracy_with_position(model_type, *out, rule_label = t_rule, position = pre_position)
        
        
    
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            break
        
    if i == MAX_iter - 1:
        return 0
    
#%%    
    out = model_size(s)
    
    for i in range(MAX_iter):
        
        choose_right, s_regression = choose_accuracy_with_position(model_size, *out, rule_label = s_rule, position = pre_position)
        
        
    
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            break
        
    if i == MAX_iter - 1:
        return 0
    
    return 1

right = 0
with tqdm(total= len(val_loader)) as pbar:
    
    for index, (_, tsc, r_tsc) in enumerate(val_loader):
        
        # print(tsc.shape)
        
        tsc = tsc.long().cuda()
        
        r_tsc = r_tsc.long().cuda()
        
        right = right + discrimnater(tsc[:,0], tsc[:,1], tsc[:,2], r_tsc[:,0], r_tsc[:,1], r_tsc[:,2])
        
        pbar.set_postfix(right_rate = right/(index + 1))#进度条
        pbar.update(1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
