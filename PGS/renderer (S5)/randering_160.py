# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:02:13 2021

@author: yuanbeiming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Blocks_clip import ViT_with_cls, take_cls, Bottleneck_judge, graph_mask_transformer, graph_transformer, Reshape, ViT_reverse_with_cls, ViT_with_cls
from einops.layers.torch import Rearrange
import numpy as np
from torch.autograd import Variable

import random

big = False
dropout = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VIC_constant(nn.Module):
    def __init__(self, attr = 'color', num_entity = 9):
        super().__init__()
        
        size = 80
        patch = 20
        
        if big:
            num_head = 4
            num_depth = 6
            self.low_dim = num_entity
            
            self.rule_dim = 10
        else:
            num_head = 4
            num_depth = 3
            self.low_dim = num_entity
            
            self.rule_dim = 10
             
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0
            
        self.attr = attr
            
        self.name = 'rander' + '_master'
        
        

        self.attribute = 20

        
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
        
        self.arrange_attr = Rearrange('b s n l -> (b n) s l', s = 3, l = self.low_dim)
        
        self.temperature = 0.01
        
        self.Embedding_t = nn.Embedding(7, self.attribute)
        
        self.Embedding_s = nn.Embedding(7, self.attribute)
        
        self.Embedding_c = nn.Embedding(11, self.attribute)
        
        self.encoder = nn.Sequential(ViT_with_cls(image_size = 160, 
                                    patch_size = 20,  
                                    dim = self.low_dim*self.attribute, 
                                    depth = num_depth, 
                                    heads = num_head, 
                                    mlp_dim = self.low_dim*self.attribute, 
                                    channels = 1, 
                                    dim_head = int(self.low_dim*self.attribute/num_head), 
                                    # dropout = 0., 
                                    # emb_dropout = 0., 
                                    num_cls = 2),
        
                                     take_cls(keepdim=True, num = 2))

        self.recon_clip = nn.Sequential(
                                          ViT_reverse_with_cls(words = 4, 
                                                               
                                                               image_size = 160,  
                                                               
                                                               patch_size = 20,
                                                               
                                                               channels = 1,
                                                               
                                                               dim = self.low_dim*self.attribute, 
                                                               
                                                                depth = num_depth*2,
                                                                
                                                                heads = num_head,
                                                                
                                                                mlp_dim = self.low_dim*self.attribute,
                                                                
                                                                dim_head = int(self.low_dim*self.attribute/num_head),
                                                                
                                          )
                                          )
        
        self.judement = nn.Sequential(
                                          ViT_with_cls(
                                                               image_size = 160,  
                                                               
                                                               patch_size = 20,
                                                               
                                                               channels = 1,
                                                               
                                                               dim = self.low_dim*self.attribute, 
                                                               
                                                                depth = num_depth*2,
                                                                
                                                                heads = num_head,
                                                                
                                                                mlp_dim = self.low_dim*self.attribute,
                                                                
                                                                dim_head = int(self.low_dim*self.attribute/num_head),
                                                                
                                                                num_cls = 3
                                                                
                                          ),
                                          
                                          take_cls(keepdim=True, num = 3)
                                          
                                          
                                          )
        
        self.T = Bottleneck_judge(self.low_dim*self.attribute, self.low_dim*self.attribute, 7*self.low_dim)
        
        self.S = Bottleneck_judge(self.low_dim*self.attribute, self.low_dim*self.attribute, 7*self.low_dim)
        
        self.C = Bottleneck_judge(self.low_dim*self.attribute, self.low_dim*self.attribute, 11*self.low_dim)
      

        

    def forward(self, state, x, ):
        b, n, h, w = state.shape
        
        b, s, n, l = x.shape

        state = state.reshape(-1, 1, h, w)
        
        flag = random.random() 
        
        if flag < 0.:
            
            mu, logvar = torch.zeros(b*n, 1, self.low_dim*self.attribute).to(x.device), torch.zeros(b*n, 1, self.low_dim*self.attribute).to(x.device)
        
            z = torch.randn(b*n, self.low_dim, self.attribute).to(x.device)
            
            # print('z')
            
        else:
            
            mu, logvar = self.encoder(state).chunk(2, dim = 1)
            
            z = self.reparametrize(mu, logvar).reshape(-1, self.low_dim, self.attribute)
        
        
        
        attr_label = x = self.arrange_attr(x)
        
        # print(attr_label.shape)
        
        # print(z.shape, x.shape)
        
        x = torch.stack([z, self.Embedding_t(x[:,0]), self.Embedding_s(x[:,1]), self.Embedding_c(x[:,2])] , dim = 1)
        
        
        recon = self.recon_clip(x.reshape(-1, 4, self.low_dim*self.attribute))
        
        recon = torch.sigmoid(recon)
        
        score = self.judement(recon)
        
        T = self.T(score[:,0]).reshape(b*n, self.low_dim, 7)
        
        S = self.S(score[:,1]).reshape(b*n, self.low_dim, 7)
        
        C = self.C(score[:,2]).reshape(b*n, self.low_dim, 11)
        
        
        return recon, state, mu, logvar, attr_label, T, S, C
    
    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))
    
        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)
        return total_kld, dimension_wise_kld, mean_kld
    
    
    
    def reconstruction_loss(self, x, x_recon, distribution='gaussian'):
        
        
        batch_size = x.size(0)
        assert batch_size != 0
    
        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'gaussian':
            #x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        else:
            recon_loss = None
    
        return recon_loss   
    

  
    def reparametrize(self, mu, logvar):
        
        
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
        
            

    
    
    @torch.no_grad()
    def generate(self, x):
        
        # x = self.arrange_attr(x)
        
        # print(z.shape, x.shape)
        
        z = torch.randn(x.shape[0], self.low_dim, self.attribute).to(x.device)
        
        x = torch.stack([z, self.Embedding_t(x[:,0]), self.Embedding_s(x[:,1]), self.Embedding_c(x[:,2])] , dim = 1)
        
        
        recon = self.recon_clip(x.reshape(-1, 4, self.low_dim*self.attribute))
        
        recon = torch.sigmoid(recon)
        

        return recon
    
        
    def loss_function(self, *out, **kwargs):
        
        attr_label, T, S, C = out[-4:]
        
        
        loss_2 = F.mse_loss(out[0], out[1], reduction='sum')
        
        loss_1 = self.kl_divergence(out[2].reshape(-1, self.low_dim*self.attribute), out[3].reshape(-1, self.low_dim*self.attribute))[-1]
        
        loss_3 = F.cross_entropy(T.reshape(-1, 7), attr_label[:,0].reshape(-1) )
        
        loss_4 = F.cross_entropy(S.reshape(-1, 7), attr_label[:,1].reshape(-1) )
        
        loss_5 = F.cross_entropy(C.reshape(-1, 11), attr_label[:,2].reshape(-1) )
        
        I_loss = loss_3 + loss_4 + loss_5
        
        # print(loss_1.shape)
        
        # loss_2 = self.reconstruction_loss(out[0], out[1])
        
        return {'loss': loss_1 + loss_2 + 1e4*I_loss, 'recon':  out[0].data, 'state': out[1].data, 'recon_loss': loss_2.item(), 
                'kld_loss': loss_1.item(), 'I(px,py)_loss': I_loss.item()}
    
    




    

if __name__ == '__main__':
    

    
    
    model = VIC_constant().to(device)
    
    x = torch.randn(1,9,80,80).to(device)
    
    y_ = torch.randint(7, (1,3,9,9)).to(device).long()
    # y_ = torch.randint(7,(5,2)).to(device)
    y = model(x, y_)
    loss = model.loss_function(*y)
    loss['loss'].backward()
    
    

    
        
        
        
