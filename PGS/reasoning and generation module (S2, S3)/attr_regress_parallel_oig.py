# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:02:13 2021

@author: yuanbeiming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Blocks_clip import ViT_with_cls, take_cls, Bottleneck_judge, graph_mask_transformer, graph_transformer, Reshape
from einops.layers.torch import Rearrange
import numpy as np


big = False
dropout = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VIC_constant(nn.Module):
    def __init__(self, attr = 'size'):
        super().__init__()
        
        size = 80
        patch = 20
        
        if big:
            num_head = 3
            num_depth = 6
            self.low_dim = 5
            
            # self.rule_dim = 10
        else:
            num_head = 3
            num_depth = 3
            self.low_dim = 5
            
            # self.rule_dim = 10
             
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0
            
        self.attr = attr
            
        self.name = 'attr_model_plus_oig' + '_' + self.attr
        
        
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
                                                            mlp_dim = self.low_dim*self.attribute, num_cls = 2, dropout = 0.1),
                                          take_cls(keepdim = True, num = 2),
                                          )
        
        self.regress_clip = nn.Sequential(
                                          graph_transformer(words = 10, dim = self.low_dim*self.attribute, 
                                                            depth = num_depth*2, heads = num_head, dim_head = int(self.low_dim*self.attribute/num_head),
                                                            mlp_dim = self.low_dim*self.attribute, num_cls = self.low_dim, dropout = 0.1),

                                          )
        
        
        self.mlp = nn.Sequential(
                                    Rearrange('b n d -> (b n) d', n = self.low_dim, d = self.low_dim*self.attribute),
                                    Bottleneck_judge(self.low_dim*self.attribute, self.low_dim*self.attribute, self.attribute),
                                    Rearrange('(b n) d -> b n d', n = self.low_dim, d = self.attribute),
                                 )

        

    def forward(self, x):
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
        
        
        
        x_regression = self.regress_clip(torch.cat((rule, x_embedding[:,:8]), dim = 1))[:,:self.low_dim]
        
        x_regression = self.mlp(x_regression)
        # 
        return x, x_regression, (rule, rule_1, rule_2), txt
  
        
        
    def loss_function(self, *out, rule_label_out, rule_label_in):
        
        x, x_regression, rule, txt = out
        
        # print(rule.shape)
        
        x = x[:,-1].long()
        
        b = x.shape[0]
        # print(x_regression.shape)
        
        # print(x.shape)
        
        txt = txt.mean(dim = 0, keepdim = True).unsqueeze(1)# b, 1, 28, dim
        
        r = F.cosine_similarity(txt, rule[0].unsqueeze(2), dim = -1)# b, 2, 1, dim
        
        r_1 = F.cosine_similarity(txt, rule[1].unsqueeze(2), dim = -1)# b, 2, 1, dim
        
        r_2 = F.cosine_similarity(txt, rule[2].unsqueeze(2), dim = -1)# b, 2, 1, dim
        

        
        loss_1_out = F.cross_entropy(r[:, 0]/self.temperature, rule_label_out.long()) + F.cross_entropy(r_1[:, 0]/self.temperature, rule_label_out.long()) + F.cross_entropy(r_2[:, 0]/self.temperature, rule_label_out.long())
        
        loss_1_in = F.cross_entropy(r[:, 1]/self.temperature, rule_label_in.long()) + F.cross_entropy(r_1[:, 1]/self.temperature, rule_label_in.long()) + F.cross_entropy(r_2[:, 1]/self.temperature, rule_label_in.long())
        
        
        loss_1 = loss_1_in + loss_1_out
        
        
        
        rule_right_out = (r[:, 0].argmax(dim = -1) == rule_label_out).float().sum() + (r_1[:, 0].argmax(dim = -1) == rule_label_out).float().sum() + (r_2[:, 0].argmax(dim = -1) == rule_label_out).float().sum()
        
        rule_right_in = (r[:, 1].argmax(dim = -1) == rule_label_in).float().sum() + (r_1[:, 1].argmax(dim = -1) == rule_label_in).float().sum() + (r_2[:, 1].argmax(dim = -1) == rule_label_in).float().sum()
        
        rule_right_out = rule_right_out/3
        
        rule_right_in = rule_right_in/3
        
        x_regression = F.gumbel_softmax(x_regression, dim = -1) + 1e-9
        
        x_regression = x_regression.log()
        
        loss_2 = F.nll_loss(x_regression.reshape(b*self.low_dim, self.attribute), x.reshape(-1))
        
        choose_right = self.choose_accuracy(*out, rule_label_out = rule_label_out, rule_label_in = rule_label_in)
        
        
        return {'loss': loss_1 + loss_2, 
                
                'loss_rule': loss_1.item() , 
                
                'loss_regression': loss_2.item(), 
                
                'rule_accuracy_out': rule_right_out, 
                
                'rule_accuracy_in': rule_right_in, 
                
                'choose_accuracy': choose_right}
    
    @torch.no_grad()
    def choose_accuracy(self, *out, rule_label_out, rule_label_in):
        
        x, x_regression, rule, txt = out
        
        rule = rule[0].unsqueeze(2)
        
        x_embedding = self.Embedding(x).reshape(-1, 9, self.low_dim*self.attribute)
        
        # print(x.shape)
        
        txt = txt.mean(dim = 0, keepdim = True).unsqueeze(1)
        
        x_regression = F.gumbel_softmax(x_regression, dim = -1).argmax(dim = -1)
        
        x_regression = self.Embedding(x_regression).reshape(-1, 1, self.low_dim*self.attribute)
        
        x_embedding = torch.cat((x_embedding[:,:8], x_regression), dim = 1)
        
        assert x_embedding.shape[1] == 9
        
        
        rule_1 = self.rule_attr_clip(x_embedding[:,3:]).unsqueeze(2)
        
        rule_2 = self.rule_attr_clip(x_embedding[:,[0,1,2,6,7,8]]).unsqueeze(2)
        
        r = F.cosine_similarity(txt, rule, dim = -1)
        
        r_1 = F.cosine_similarity(txt, rule_1, dim = -1)
        
        r_2 = F.cosine_similarity(txt, rule_2, dim = -1)
        
        rule_right_out = (r[:, 0].argmax(dim = -1) == rule_label_out).float() + (r_1[:, 0].argmax(dim = -1) == rule_label_out).float() + (r_2[:, 0].argmax(dim = -1) == rule_label_out).float()
        
        
        rule_right_in = (r[:, 1].argmax(dim = -1) == rule_label_in).float() + (r_1[:, 1].argmax(dim = -1) == rule_label_in).float() + (r_2[:, 1].argmax(dim = -1) == rule_label_in).float()
        
        
        
        r = rule_right_out + rule_right_in

        return (r.eq(6)).float().sum()




    

if __name__ == '__main__':
    model = VIC_constant().to(device)
    
    x = torch.randint(7, (5,9,5)).cuda()
    
    y_ = torch.randint(28, (5,)).cuda().float()
    # y_ = torch.randint(7,(5,2)).to(device)
    y = model(x.long())
    loss = model.loss_function(*y, rule_label_out = y_, rule_label_in = y_)
    loss['loss'].backward()
    
    

    
        
        
        
