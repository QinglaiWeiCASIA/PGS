# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm


from attr_regress_with_rule import attr_regress_plus_all_model_lr as model_vic


from data_aug import transform_01, reverse_transform_01


import numpy as np
from torchvision import transforms



t = transform_01

import make_generate_attribute_oig as make_data




    
if make_data.dataset == 'left_center_single_right_center_single':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model_lr as model_vic
    
    
elif make_data.dataset == 'up_center_single_down_center_single':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model_lr as model_vic
    
elif make_data.dataset == 'in_distribute_four_out_center_single':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model_oig as model_vic
    
    
elif make_data.dataset == 'in_center_single_out_center_single':
    
    
    from attr_regress_with_rule import attr_regress_plus_all_model_oic as model_vic

else:
    
    print('dataset error')

batch_size = 160


weight_decay = 0


train_set = make_data.Raven_Data(train = True,val = False)
len_train_set = len(train_set)

print(len_train_set)



train_loader , num_train = make_data.make_loader(train_set, batch_size)#3346,2529
print(num_train, len(train_loader))



num_train = len(train_set)


print(num_train, len(train_loader))

val_loader, num_val = make_data.raven_loader(batch_size, train = False, val = True)


print('train:', num_train, 'test:', num_val)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model_vic.VIC_constant(attr = make_data.attribute_)

if model_vic.big == True:
	name =  'big_' + model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  'big_' + model.name  + make_data.aaa
else:
	name =  model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  model.name +  make_data.aaa
print(name)
import os

	
#model.load_state_dict(torch.load('./model_'+name+'_now.pt', map_location = 'cpu'))
# /home/yuanbeiming/python_work/vit_for_raven/vit_for_raven_92.pt
#%%
if torch.cuda.device_count() > 1:



  model = nn.DataParallel(model)
  print( torch.cuda.device_count())

model = model.to(device)


#%%

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
#optimiser.load_state_dict(torch.load('./optimiser_'+name+'_now.pt'))

for param_group in optimiser.param_groups:
    #param_group["lr"] = 7e-4
    param_group["weight_decay"] = weight_decay

print('lr:', optimiser.state_dict()['param_groups'][0]['lr'])
print('w_d:', optimiser.state_dict()['param_groups'][0]['weight_decay'])

lr_decay = torch.optim.lr_scheduler.StepLR(optimiser, step_size= 1, gamma= 0.99)



#%%
min_loss = 1e9
max_acc = 0
num_epoch = 120000
epoch = 0
save_and_sample_every = 1

#%%
with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        f.write('train_num_sample:' + str(len_train_set)+ '\n')
        # f.write('temperature:' + str(model_vit.temperature)+ '\n')
        f.write('weight_decay:' + str(weight_decay)+ '\n')

# print('temperature:',model_vit.temperature)
print('weight_decay:' ,(weight_decay))

while epoch < num_epoch:
    accuracy = [0]*3
    
    loss_train = [0]*4
    
    loss_test_all = [0]*4

    
    accuracy_val = [0]*3
    

    # loss_test_all = 0
    with tqdm(total=len(train_loader)  + len(val_loader)) as pbar:
        model.train()  #启用 Batch Normalization 和 Dropout。
        for _, x_train, label_out, label_in in train_loader:


            # Model training
           
            
            #梯度清零

            x_train = (x_train).long().to(device)
            #print(label[:2])


         
            
            label_out = label_out.float().to(device)
            
            label_in = label_in.float().to(device)




            out_train = model(x_train) #输出

            

            loss_out = (model.module.loss_function(*out_train, rule_label_out = label_out, rule_label_in = label_in) if isinstance(model, nn.DataParallel) 
                    else model.loss_function(*out_train, rule_label_out = label_out, rule_label_in = label_in))

            loss = loss_out['loss']
            
            loss_train[0] += loss.item()
            
            loss_train[1] += loss_out['loss_rule']
            
            loss_train[2] += loss_out['loss_regression']
            

            
            accuracy[0] += loss_out['choose_accuracy']
            
            accuracy[1] += loss_out['rule_accuracy_out']
            
            accuracy[2] += loss_out['rule_accuracy_in']
            
            
            model.zero_grad()

            loss.backward()#回传

            optimiser.step()  
            

            pbar.set_postfix(loss_batch = loss.item())#进度条
            pbar.update(1)
        lr_decay.step()
        
        # accuracy[0] /= (num_train*8)
        accuracy[0] /= (num_train)
        
        accuracy[1] /= (num_train)
        
        accuracy[2] /= (num_train)
        
        for i in range(4):
                
                loss_train[i] /= len(train_loader)

        
        model.eval()#不启用 Batch Normalization 和 Dropout。
        with torch.no_grad():
            
            for index, (_, x_test, label_out, label_in)in enumerate(val_loader):
 
                x_test = (x_test).long().to(device)

                label_out = label_out.float().to(device)
                
                label_in = label_in.float().to(device)
                
                out_test = model(x_test)
                

                loss_out = (model.module.loss_function(*out_test, rule_label_out = label_out, rule_label_in = label_in) if isinstance(model, nn.DataParallel) 
                        else model.loss_function(*out_test, rule_label_out = label_out, rule_label_in = label_in))

                
                loss_test_all[0] += loss_out['loss'].item()
                
                loss_test_all[1] += loss_out['loss_rule']
                
                loss_test_all[2] += loss_out['loss_regression']
                

                
                accuracy_val[0] += loss_out['choose_accuracy']
                
                accuracy_val[1] += loss_out['rule_accuracy_out']
                
                accuracy_val[2] += loss_out['rule_accuracy_in']

                
                pbar.set_postfix(loss_batch = loss_out['loss'].item())
                pbar.update(1)


            accuracy_val[0] /= (num_val)
            
            accuracy_val[1] /= (num_val)
            
            accuracy_val[2] /= (num_val)
            
            for i in range(4):
                loss_test_all[i] /= len(val_loader)

            

    
    # Stores model
    if accuracy_val[0] > max_acc: 
        torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_best.pt')
        torch.save(optimiser.state_dict(), './optimiser_'+name+'_best.pt')

        max_acc = accuracy_val[0]
        

    torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_now.pt')
    torch.save(optimiser.state_dict(), './optimiser_'+name+'_now.pt')

    # Print and log some results
    print("epoch:{}\n loss_train: all: {:.4f}\t  rule:  {:.4f}\t  regression {:.4f}\t  choose_accuracy:  {:.4f}\t  rule_accuracy: out: {:.4f}\t in: {:.4f}\n loss_test:  all: {:.4f}\t  rule:  {:.4f}\t  regression {:.4f}\t  choose_accuracy:  {:.4f}\t  rule_accuracy: out: {:.4f}\t in: {:.4f}\n  learning_rate:{:.8f}\n".\
          format(epoch, loss_train[0], loss_train[1],  loss_train[2], accuracy[0], accuracy[1], accuracy[2], loss_test_all[0], loss_test_all[1],  loss_test_all[2],  accuracy_val[0], accuracy_val[1], accuracy_val[2], lr_decay.get_lr()[0]))

    with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        
        f.write("epoch:{}\n loss_train: all: {:.4f}\t  rule:  {:.4f}\t  regression {:.4f}\t  choose_accuracy:  {:.4f}\t  rule_accuracy: out: {:.4f}\t in: {:.4f}\n loss_test:  all: {:.4f}\t  rule:  {:.4f}\t  regression {:.4f}\t  choose_accuracy:  {:.4f}\t  rule_accuracy: out: {:.4f}\t in: {:.4f}\n  learning_rate:{:.8f}\n".\
              format(epoch, loss_train[0], loss_train[1],  loss_train[2], accuracy[0], accuracy[1], accuracy[2], loss_test_all[0], loss_test_all[1],  loss_test_all[2],  accuracy_val[0], accuracy_val[1], accuracy_val[2], lr_decay.get_lr()[0]))
    epoch += 1
    
    
    #%%











    
# from torchvision import transforms
# from matplotlib import pyplot as plt  
# t = transforms.ToPILImage()
# from einops.layers.torch import Rearrange
# arrenge = Rearrange('c h w -> h (c w)')
# num = 6
# plt.imshow(t(arrenge(x_train[num][:3])), cmap = 'gray')
# plt.show() 


# num = 6

# for i in range(8):

#     plt.subplot(3,3,i+1)
    
#     plt.imshow(t(x_train[num][i]), cmap = 'gray')
# plt.show() 
    


# for i in range(8):
    
#     plt.subplot(2,4,i+1)
    
#     plt.imshow(t(x_train[num][i+8]), cmap = 'gray')

    
    
# plt.show()

# print(y_train[num])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
