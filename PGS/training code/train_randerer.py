# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm


import randering_160 as model_vic


from data_aug import transform_01, reverse_transform_01, reverse_PIL


import numpy as np
from torchvision import transforms



t = transform_01

import make_generate_all_attribute as make_data
# import make_generate_all_attribute_oig as make_data



if make_data.dataset == 'distribute_four':

    num_entity = 4
    

elif make_data.dataset == 'distribute_nine':
    
    
    num_entity = 9
    
    
elif make_data.dataset == 'center_single':
    
    
    num_entity = 1


    
elif make_data.dataset == 'left_center_single_right_center_single':
    
    
    num_entity = 2
    
    
elif make_data.dataset == 'up_center_single_down_center_single':
    
    
    num_entity = 2
    
elif make_data.dataset == 'in_distribute_four_out_center_single':
    
    
    num_entity = 5
    
    
elif make_data.dataset == 'in_center_single_out_center_single':
    
    
    num_entity = 2

else:
    
    print('dataset error')

batch_size = 80


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

model = model_vic.VIC_constant(num_entity = num_entity)

if model_vic.big == True:
	name =  'big_' + model.name +  '_' +str(len_train_set) + '_' + make_data.aaa+'_configure_160'
	text_name =  'big_' + model.name  + make_data.aaa
else:
	name =  model.name +  '_' +str(len_train_set) + '_' + make_data.aaa+'_configure_160'
	text_name =  model.name +  make_data.aaa
print(name)
import os

num_entity = model.low_dim	



result_path = make_data.dataset + '_rander_result'

if not os.path.exists(result_path):
    
    os.mkdir(result_path)
model.load_state_dict(torch.load('./model_'+name+'_now.pt', map_location = 'cpu'))
# /home/yuanbeiming/python_work/vit_for_raven/vit_for_raven_92.pt
#%%
if torch.cuda.device_count() > 1:



  model = nn.DataParallel(model)
  print( torch.cuda.device_count())

model = model.to(device)



#%%

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
optimiser.load_state_dict(torch.load('./optimiser_'+name+'_now.pt'))

for param_group in optimiser.param_groups:
    param_group["lr"] = 5e-6
    param_group["weight_decay"] = weight_decay

print('lr:', optimiser.state_dict()['param_groups'][0]['lr'])
print('w_d:', optimiser.state_dict()['param_groups'][0]['weight_decay'])

lr_decay = torch.optim.lr_scheduler.StepLR(optimiser, step_size= 1, gamma= 0.995)



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
        for x_train, label,  *_ in train_loader:


            # Model training
           
            
            #梯度清零

            x_train = (x_train/255.).float().to(device)
            #print(x_train.shape)

         
            
            label = label.long().to(device)
            #print(label[0, :, 0])




            out_train = model(x_train, label) #输出
 

            

            loss_out = (model.module.loss_function(*out_train, rule_label = label) if isinstance(model, nn.DataParallel) 
                    else model.loss_function(*out_train, rule_label = label))

            loss = loss_out['loss']
            
            loss_train[0] += loss.item()
            
            loss_train[1] += loss_out['recon_loss']
            
            loss_train[2] += loss_out['kld_loss']
            
            loss_train[3] += loss_out['I(px,py)_loss']
            

            
            # accuracy[0] += loss_out['choose_accuracy']
            
            # accuracy[1] += loss_out['rule_accuracy']
            
            
            model.zero_grad()

            loss.backward()#回传

            optimiser.step()  
            

            pbar.set_postfix(loss_batch = loss.item())#进度条
            pbar.update(1)
            
            # break
        lr_decay.step()
        
        # accuracy[0] /= (num_train*8)
        # accuracy[0] /= (num_train)
        
        # accuracy[1] /= (num_train)
        
        for i in range(4):
                
                loss_train[i] /= len(train_loader)

        
        model.eval()#不启用 Batch Normalization 和 Dropout。
        with torch.no_grad():
            
            for index, ( x_test, label,  *_)in enumerate(val_loader):
 
                x_test = (x_test/255.).float().to(device)

                label = label.long().to(device)
                #idx = (idx).long().to(device)
                
                out_test = model(x_test, label)
                

                loss_out = (model.module.loss_function(*out_test, rule_label = label) if isinstance(model, nn.DataParallel) 
                        else model.loss_function(*out_test, rule_label = label))

                
                loss_test_all[0] += loss_out['loss'].item()
                
                loss_test_all[1] += loss_out['recon_loss']
                
                loss_test_all[2] += loss_out['kld_loss']
                
    
                
                loss_test_all[3] += loss_out['I(px,py)_loss']
                
                # accuracy_val[1] += loss_out['rule_accuracy']

                
                pbar.set_postfix(loss_batch = loss_out['loss'].item())
                pbar.update(1)
                # break
                


            # accuracy_val[0] /= (num_val)
            
            # accuracy_val[1] /= (num_val)
            
            for i in range(4):
                loss_test_all[i] /= len(val_loader)
                
            
            
            step = epoch

            if step != 10000 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every

                

                
                

                num = 5
                #batches = num_to_groups(1, batch_size)
                #all_images_list = list(map(lambda n: sample(model, image_size = 80, batch_size=n, channels=1), batches))
                
                re_sample = 1

                index = torch.randperm(x_test.shape[0]*9)

                
                all_images = [loss_out['state'][index[:num]]]
                
                input_data = label.permute(0,2,1,3).reshape(-1, 3, num_entity)[index[:num]]
                
                all_images.append(model.module.generate(input_data))
               
                

                    
                all_images = torch.stack(all_images, dim=2).permute(0,1,3,2,4).reshape(-1,1,160,int(160*(re_sample + 1))).detach().cpu()
                for i in range(num):
                	images_ = reverse_transform_01(all_images[i])
                	images_.save("./"+result_path +"/image_"+ str(epoch)+ "_"+ str(i) +".jpeg")
                    


            

    
    # Stores model
    if accuracy_val[0] > max_acc: 
        torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_best.pt')
        torch.save(optimiser.state_dict(), './optimiser_'+name+'_best.pt')

        max_acc = accuracy_val[0]

    
        

    torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_now.pt')
    torch.save(optimiser.state_dict(), './optimiser_'+name+'_now.pt')

    # Print and log some results
    print("epoch:{}\n loss_train: all: {:.4f}\t  recon_loss: {:.4f}\t kld_loss: {:.4f}\t I(pxpy): {:.4f}\t loss_test:  all: {:.4f}\t  recon_loss: {:.4f}\t kld_loss: {:.4f}\t  I(pxpy): {:.4f}\t learning_rate:{:.8f}\n".\
          format(epoch, loss_train[0], loss_train[1], loss_train[2],  loss_train[3], loss_test_all[0], loss_test_all[1], loss_test_all[2], loss_train[3], lr_decay.get_lr()[0]))

    with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        
        f.write("epoch:{}\n loss_train: all: {:.4f}\t   loss_test:  all: {:.4f}\t  learning_rate:{:.8f}\n".\
              format(epoch, loss_train[0], loss_test_all[0], lr_decay.get_lr()[0]))
    epoch += 1
    
    
    if epoch == 2:
        for param_group in optimiser.param_groups:
    	    param_group["lr"] = 1e-4
    	    #param_group["weight_decay"] = weight_decay
        
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
