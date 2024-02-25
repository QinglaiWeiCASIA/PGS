#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:09:30 2023

@author: yuanbeiming
"""
# from attr_regress_with_rule import attr_regress_decoder_mask as model_vic

import make_generate_all_attribute as make_data


if make_data.dataset == 'distribute_four':

    from attr_regress_with_rule import attr_regress_decoder_mask_d4 as model_vic
    

elif make_data.dataset == 'distribute_nine':
    
    
    from attr_regress_with_rule import attr_regress_decoder_mask as model_vic
    

else:
    
    print('dataset error')

import torch

import torch.nn.functional as F

from tqdm import tqdm

dataset = make_data.dataset

model_color = model_vic.VIC_constant(attr = 'color')

model_color.load_state_dict(torch.load('./model_'+model_color.name+'_'+str(len(make_data.train_file))+'_Datasets_'+dataset+'_color_now.pt', map_location = 'cpu'))

model_type = model_vic.VIC_constant(attr = 'type')

model_type.load_state_dict(torch.load('./model_'+model_type.name+'_'+str(len(make_data.train_file))+'_Datasets_'+dataset+'_type_now.pt', map_location = 'cpu'))

model_size = model_vic.VIC_constant(attr = 'size')

model_size.load_state_dict(torch.load('./model_'+model_size.name+'_'+str(len(make_data.train_file))+'_120000_Datasets_'+dataset+'_size_now.pt', map_location = 'cpu'))

model_color.cuda()

model_type.cuda()

model_size.cuda()

model_color.eval()

model_type.eval()

model_size.eval()

val_loader, num_val = make_data.raven_loader(1, train = False, val = False)

kv = torch.Tensor([[11,12,12,12,12,12,12,12,12,12]]).cuda().long()

def choose_accuracy(self, *out, rule_label):
    
    x, _, rule, txt = out
    
    
    x_regression = self.decode_attr(x)
    attr_regression = x_regression
    # print(x_regression.shape)
    
    rule = rule[0]
    
    x_embedding = self.Embedding(x).reshape(-1, 9, self.low_dim*self.attribute)
    
    # print(x.shape)
    
    txt = txt.mean(dim = 0, keepdim = True)
    
    # x_regression = F.gumbel_softmax(x_regression, dim = -1).argmax(dim = -1)
    
    x_regression = self.Embedding(x_regression).reshape(-1, 1, self.low_dim*self.attribute)
    
    x_embedding = torch.cat((x_embedding[:,:8], x_regression), dim = 1)
    
    assert x_embedding.shape[1] == 9
    
    
    rule_1 = self.rule_attr_clip(x_embedding[:,3:])
    
    rule_2 = self.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]])
    
    r = F.cosine_similarity(txt, rule, dim = -1)
    
    r_1 = F.cosine_similarity(txt, rule_1, dim = -1)
    
    r_2 = F.cosine_similarity(txt, rule_2, dim = -1)
    
    r = (r.argmax(dim = -1) == rule_label).float()
    
    r_1 = (r_1.argmax(dim = -1) == rule_label).float()
    
    r_2 = (r_2.argmax(dim = -1) == rule_label).float()
    
    
    
    r = r + r_1 + r_2
    
    return (r.eq(3)).float().sum(), attr_regression


    
def choose_accuracy_with_position(self, *out, rule_label, position):
    
    x, _, rule, txt = out
    
    
    #x_regression = self.decode_attr_with_position(x, position)

    x_regression = self.decode_attr(x)
    # print(x_regression.shape)
    
    x_regression = (x_regression*position + (1-position)*(self.attribute - 1)).long()
    
    attr_regression = x_regression
    
    rule = rule[0]
    
    x_embedding = self.Embedding(x).reshape(-1, 9, self.low_dim*self.attribute)
    
    # print(x.shape)
    
    txt = txt.mean(dim = 0, keepdim = True)
    
    # x_regression = F.gumbel_softmax(x_regression, dim = -1).argmax(dim = -1)
    
    x_regression = self.Embedding(x_regression).reshape(-1, 1, self.low_dim*self.attribute)
    
    x_embedding = torch.cat((x_embedding[:,:8], x_regression), dim = 1)
    
    assert x_embedding.shape[1] == 9
    
    
    rule_1 = self.rule_attr_clip(x_embedding[:,3:])
    
    rule_2 = self.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]])
    
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
    
    out = model_color(c, kv)
    
    for i in range(MAX_iter):
        
        choose_right, c_regression = choose_accuracy(model_color, *out, rule_label = c_rule)
        
        
        
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            print(c_regression)
            break
        
    if i == MAX_iter - 1:
        return 0
            
            
    pre_position = (c_regression != 10).float()
    assert pre_position.shape[1] == 9
#%%
    out = model_type(t, kv)
    
    for i in range(MAX_iter):
        
        choose_right, t_regression = choose_accuracy_with_position(model_type, *out, rule_label = t_rule, position = pre_position)
        
        
    
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            print(t_regression)
            break
        
    if i == MAX_iter - 1:
        return 0
    
#%%    
    out = model_size(s, kv)
    
    for i in range(MAX_iter):
        
        choose_right, s_regression = choose_accuracy_with_position(model_size, *out, rule_label = s_rule, position = pre_position)
        
        
        
    
        if choose_right.item() == 0:
            
            continue
        
        else:
            print(choose_right)
            print(s_regression)
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
