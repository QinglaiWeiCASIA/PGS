# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:02:13 2021

@author: yuanbeiming
"""

import torch
import resnet
import torch.nn as nn

__all__ = ['my_cmp']
class cmp_learning(nn.Module):
    def __init__(self):
         super(cmp_learning,self).__init__()
         self.resnet = resnet.ResNet50(100)
         self.pre_head = nn.Sequential(nn.ReLU(), nn.Linear(100, 1))
         self.mseloss = nn.MSELoss()
         # self.ear = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=100, kernel_size=10, stride=3, bias=False),
         #                          nn.BatchNorm2d(100),
         #                          nn.ReLU(inplace=True),
         #                          nn.AdaptiveAvgPool2d(1),
         #                          nn.BatchNorm2d(100))
         self.ear = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=100, kernel_size=160, stride=3, bias=False),
                                  nn.BatchNorm2d(100),
                                  nn.ReLU(inplace=True),
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.BatchNorm2d(100))
    # def forward(self, x):
    #     x = x[:,:3,...]
    #     x = x.reshape(-1, 1, 160, 160)
    #     residual = self.ear(x).reshape(-1,100)
    #     x = self.resnet(x)
    #     x_ = torch.cat((x, residual), dim = -1)#不用就关掉
    #     x_ = self.pre_head(x_)
    #     x_ = x_.reshape(-1, 3)
    #     return x_
    # def forward(self, x):
    #     x = x[:,:3,...]
    #     x = x.reshape(-1, 1, 160, 160)
    #     # residual = self.ear(x).reshape(-1,100)
    #     x = self.resnet(x)
    #     # x_ = torch.cat((x, residual), dim = -1)#不用就关掉
    #     x_ = self.pre_head(x)
    #     x_ = x_.reshape(-1, 3)
    #     return x_
    def forward(self, x):
        x = x[:,:3,...]
        x = x.reshape(-1, 1, 160, 160)
        residual = self.ear(x).reshape(-1,100)
        x = self.resnet(x)
        x = x + residual#不用就关掉
        x = self.pre_head(x)
        x = x.reshape(-1, 3)
        return x
    
    # def forward(self, x):
    #     x = x[:,:3,...]
    #     x = x.reshape(-1, 1, 160, 160)
    #     residual = self.ear(x).reshape(-1,100)
    #     x = self.resnet(x)
    #     x = x + residual#不用就关掉
    #     x = self.pre_head(x)
    #     x = x.reshape(-1, 3)
    #     return x
    
    def var(self, x, x_):
        n = x.shape[0]
        return ((x - x_)**2).sum(dim = 0)/(n-1)#3
        
    def hinge(self, x, e = 1e-8, gamma = 1):
        x = (x + e)**(1/2)
        return torch.mean(torch.clamp(gamma - x, min=0))
    
    
    def loss_function(self, x):
        loss_1 = self.mseloss(x[:,0,...], x[:,1,...].detach()) + self.mseloss(x[:,1,...], x[:,2,...].detach()) + self.mseloss(x[:,2,...], x[:,0,...].detach())
        x_ = x.reshape(-1,3).mean(dim = 0, keepdim = True)#1,3
        var = self.var(x, x_)
        loss_2 = self.hinge(var)
        return loss_1 + 200*loss_2
    
def my_cmp():
    return cmp_learning()
if __name__ == '__main__':
    model = my_cmp()
    
    x = torch.randn(2,3,160,160)
    y = model(x)
    loss = model.loss_function(y)
    loss.backward()
    
        
        
        
