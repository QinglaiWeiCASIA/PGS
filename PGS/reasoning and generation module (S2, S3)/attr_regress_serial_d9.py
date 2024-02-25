# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:02:13 2021

@author: yuanbeiming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Blocks_clip import ViT_with_cls, take_cls, Bottleneck_judge, graph_mask_transformer, graph_transformer, Reshape, Mask_Transformer_Decoder
from einops.layers.torch import Rearrange
import numpy as np


big = False
dropout = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VIC_constant(nn.Module):
    def __init__(self, attr = 'color'):
        super().__init__()
        
        size = 80
        patch = 20
        
        if big:
            num_head = 3
            num_depth = 6
            self.low_dim = 9
            
            self.rule_dim = 10
        else:
            num_head = 3
            num_depth = 3
            self.low_dim = 9
            
            self.rule_dim = 10
             
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0
            
        self.attr = attr
            
        self.name = 'attr_decoder_mask_' + '_' + self.attr
        
        
        if self.attr == 'type':
            self.attribute = 7
            
        elif self.attr == 'size':
            self.attribute = 7
            
        elif self.attr == 'color':
            self.attribute = 11
            
        else:
            assert 1 == 2
        
        self.out_dim = 1
        
        self.mseloss = nn.MSELoss()
        
        txt_data = []

        keep_dir = './'


        for A in range(6,10):#T年
                    for n in range(7,10):
                         txt_data.append(np.array([A,  1, n, 2, 0]))
                         
        for A in range(6,10):#T年
                    for n in range(7,10):
                         txt_data.append(np.array([A,  1, 0, 2, n]))
                         
        for A in range(6,10):#T年
                         txt_data.append(np.array([A,  1, 6, 2, 6]))                 
                        
        txt_data = np.array(txt_data)

        assert txt_data.shape[0] == 28
        
        
        self.txt_data = torch.Tensor(txt_data)
        
        self.temperature = 0.01
        
        self.Embedding = nn.Embedding(self.attribute, self.attribute)

        if self.attr == 'color' or self.attr == 'size':
        
        	self.kv_Embedding = nn.Embedding(13, self.low_dim*self.attribute) # start: 11, padding: 12
        
        self.kv = torch.Tensor([[self.attribute] + [self.attribute + 1]*9])

        self.rule_txt_clip = nn.Sequential(
                                          graph_mask_transformer(dict_size = 10, words = 5, dim = self.low_dim*self.attribute, 
                                                                 depth = num_depth*2, heads = num_head, dim_head = int(self.low_dim*self.attribute/num_head), 
                                                                 mlp_dim = self.low_dim*self.attribute, dropout = 0.1),
                                          take_cls(),
                                          # nn.Linear(self.rule_dim, self.low_dim)
                                          )
        
        self.rule_attr_clip = nn.Sequential(
                                          graph_transformer(words = 6, dim = self.low_dim*self.attribute, 
                                                            depth = num_depth*2, heads = num_head, dim_head = int(self.low_dim*self.attribute /num_head),
                                                            mlp_dim = self.low_dim*self.attribute, dropout = 0.1),
                                          take_cls(keepdim = True),
                                          )
        
        self.regress_clip = nn.Sequential(
                                          graph_transformer(words = 9, dim = self.low_dim*self.attribute, 
                                                            depth = num_depth*2, heads = num_head, dim_head = int(self.low_dim*self.attribute/num_head),
                                                            mlp_dim = self.low_dim*self.attribute, num_cls = 1, dropout = 0.1),

                                          )
        
        self.decoder = Mask_Transformer_Decoder(dict_size = 13, 
                                                words = 10,
                                           dim = self.low_dim*self.attribute,
                                           depth = num_depth*2, 
                                           heads = num_head,
                                           dim_head = int(self.low_dim*self.attribute/num_head),
                                           mlp_dim = self.low_dim*self.attribute,
                                           mask_data = self.attribute + 1)
        
        
        self.mlp = nn.Sequential(
                                    Rearrange('b n d -> (b n) d', n = 1, d = self.low_dim*self.attribute),
                                    Bottleneck_judge(self.low_dim*self.attribute, self.low_dim*self.attribute, self.attribute),
                                    Rearrange('(b n) d -> b n d', n = 1, d = self.attribute),
                                 )

        

    def forward(self, x, kv):
        
        # print(x.device)
        
        # print(kv.device)
        b, n, l = x.shape
        assert n == 9
        
        txt = self.txt_data.to(x.device).long()
        
        txt = self.rule_txt_clip(txt).unsqueeze(0)
        
        x_embedding = self.Embedding(x).reshape(b, 9, self.low_dim*self.attribute)
        
        # print(x_embedding.shape)
        
        # x_embedding = x_embedding
        
        rule = self.rule_attr_clip(x_embedding[:,:6])
        
        rule_1 = self.rule_attr_clip(x_embedding[:,3:])
        
        rule_2 = self.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]])
        
        
        
        x_regression = self.regress_clip(torch.cat((rule, x_embedding[:,:8]), dim = 1))[:,:1]
        
        # print(x_regression.shape)
        
        
        # kv = self.kv_Embedding(kv).reshape(b, 10, self.low_dim*self.attribute)
        # print(kv.shape)
        
        x_regression = self.decoder(kv, x_regression) #b, 1, d
        
        
        
        x_regression = self.mlp(x_regression)
        # 
        return x, x_regression, (rule, rule_1, rule_2), txt
  
        
        
    def loss_function(self, *out, rule_label, i):
        
        target, _, x_regression, rule, txt = out
        

        
        b = target.shape[0]

        
        txt = txt.mean(dim = 0, keepdim = True)
        
        r = F.cosine_similarity(txt, rule[0], dim = -1)
        
        # print(r.shape)
        
        r_1 = F.cosine_similarity(txt, rule[1], dim = -1)
        
        r_2 = F.cosine_similarity(txt, rule[2], dim = -1)
        
        # print(r.shape)
        
        loss_1 = F.cross_entropy(r/self.temperature, rule_label.long()) + F.cross_entropy(r_1/self.temperature, rule_label.long()) + F.cross_entropy(r_2/self.temperature, rule_label.long())
        
        
        if i == 8:
            rule_right = (r.argmax(dim = -1) == rule_label).float().sum() + (r_1.argmax(dim = -1) == rule_label).float().sum() + (r_2.argmax(dim = -1) == rule_label).float().sum()
            
            rule_right = rule_right/3
            
        else:
            
            rule_right = torch.zeros(1).to(x_regression.device)
        
        # if self.training:
        
        x_regression = F.gumbel_softmax(x_regression, dim = -1) + 1e-9
            
        # # else:
            
        #     x_regression = F.softmax(x_regression, dim = -1) + 1e-9
        
        x_regression = x_regression.log()
        
        loss_2 = F.nll_loss(x_regression.reshape(b, self.attribute), target.reshape(-1))
        
        if i == 8:
        
            choose_right = self.choose_accuracy( *out[1:], rule_label = rule_label)
            
        else:
            choose_right = torch.zeros(1).to(x_regression.device)
            
            
        return {'loss': loss_1 + loss_2, 'loss_rule': loss_1.item() , 'loss_regression': loss_2.item(), 'rule_accuracy': rule_right, 'choose_accuracy': choose_right}
    
    @torch.no_grad()
    def choose_accuracy(self, *out, rule_label):
        
        x, _, rule, txt = out
        
        
        x_regression = self.decode_attr(x)
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
        
        return (r.eq(3)).float().sum()
    
    
    @torch.no_grad()


    def decode_attr_with_position(self, x, position = None):
        
        
        b = x.shape[0]
        
        x = x[:,:8]
        x_embedding = self.Embedding(x).reshape(b, 8, self.low_dim*self.attribute)

        
        rule = self.rule_attr_clip(x_embedding[:,:6])
        
        
        x_regression = self.regress_clip(torch.cat((rule, x_embedding[:,:8]), dim = 1))[:,:1]
        
        
        # kv = torch.ones(b, 10).to(x.device)
        
        # kv = kv*12
        
        # kv[:,0:1] = torch.ones(b, 1).to(x.device)*11
        
        # kv = kv.long()
        
        
        
        kv = torch.Tensor([[11] + [12]*self.low_dim]).repeat(b, 1).to(device).long()
        # print(kv)
        
        for i in range(1,self.low_dim + 1):

            # kv_embedding = self.kv_Embedding(kv).reshape(b, 10, self.low_dim*self.attribute)
            
            x_attr = self.decoder(kv, x_regression) #b, 1, d
            
            x_attr = self.mlp(x_attr)#b,1, attr
            # print(x_attr.shape)
            if position is not None:
                # print(position.shape)
            
                mask = torch.Tensor([[[0]*(self.attribute-1) + [1]]]).repeat(x_attr.shape[0],1, 1).to(position.device)
                
                position_now = position[:,None,i-1:i] 
                
                mask = 1 - position_now - mask
                
                mask = mask.abs().bool()
                
                # print(mask*position_now.to(mask.device))
                
                # print(position_now.shape)
                
                x_attr.masked_fill_(mask.to(x_attr.device), -1e9)
                
                # print(x_attr)
            
            x_attr = F.gumbel_softmax(x_attr, dim = -1).argmax(dim = -1) #b, 1
            
            print(x_attr)

            kv[:,i:i+1] = x_attr
            
            # print(kv[0])
        
        return kv[:, 1:]

    @torch.no_grad()
    def decode_attr(self, x):
        
        
        b = x.shape[0]
        
        x = x[:,:8]
        x_embedding = self.Embedding(x).reshape(b, 8, self.low_dim*self.attribute)

        
        rule = self.rule_attr_clip(x_embedding[:,:6])
        
        
        x_regression = self.regress_clip(torch.cat((rule, x_embedding[:,:8]), dim = 1))[:,:1]
        
        
        # kv = torch.ones(b, 10).to(x.device)
        
        # kv = kv*12
        
        # kv[:,0:1] = torch.ones(b, 1).to(x.device)*11
        
        # kv = kv.long()
        
        
        
        kv = torch.Tensor([[11,12,12,12,12,12,12,12,12,12]]).repeat(b, 1).to(device).long()
        # print(kv)
        
        for i in range(1,10):

            # kv_embedding = self.kv_Embedding(kv).reshape(b, 10, self.low_dim*self.attribute)
            
            x_attr = self.decoder(kv, x_regression) #b, 1, d
            
            x_attr = self.mlp(x_attr)#b,1, attr
            
            x_attr = F.gumbel_softmax(x_attr, dim = -1).argmax(dim = -1) #b, 1

            kv[:,i:i+1] = x_attr
            
            # print(kv[0])
        
        return kv[:, 1:]

        




    

if __name__ == '__main__':
    model = VIC_constant().to(device)
    
    x = torch.randint(7, (5,9,9)).cuda()
    
    y_ = torch.randint(28, (5,)).cuda().float()
    
    kv = torch.Tensor([[11,12,12,12,12,12,12,12,12,12]]).repeat(5, 1).to(device).long()
    # y_ = torch.randint(7,(5,2)).to(device)
    
    # for i in range(9):
    # y = model(x.long(), kv)
    # # kv[:, i + 1] = x[:, -1, i]
    # loss = model.loss_function(x[:, -1, 0], *y,rule_label = y_)
    # model.zero_grad()
    # loss['loss'].backward()
    
    # kv = model.decode_attr(x)
    #%%
    for i in range(9):
        print(i)
        y = model(x.long(), kv.data)
        kv[:, i + 1] = x[:, -1, i].data
        loss = model.loss_function(x[:, -1, 0], *y,rule_label = y_, i = i)
        print(loss['loss'])
        model.zero_grad()
        loss['loss'].backward()
    
    kv = model.decode_attr(x)
    
    

    
        
        
        
