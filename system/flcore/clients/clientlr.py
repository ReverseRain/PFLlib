import copy
import torch.nn as nn
import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize 
from flcore.clients.clientbase import Client

class LoRAParametrization(nn.Module):  
    def __init__(self, features_in, features_out, rank=3, alpha=1, device='cpu'):  
        super().__init__()  
        # 论文第4.1节：  
        # 我们使用随机高斯初始化A，并将B初始化为零，所以ΔW = BA在训练开始时为零  
        self.lora_A = nn.Parameter(torch.zeros((rank, features_in)).to(device))  # 初始化为零的低秩矩阵A  
        self.lora_B = nn.Parameter(torch.zeros((features_out, rank)).to(device))  # 初始化为零的低秩矩阵B  
        nn.init.normal_(self.lora_A, mean=0, std=1)  # 对A进行标准正态分布初始化  
  
        self.scale = alpha / rank  # 缩放因子  
  
    def forward(self, original_weights): 
        return original_weights + torch.matmul(self.lora_B, self.lora_A) * self.scale  

class clientLR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda

        features_in=self.model.head.in_features
        features_out=self.model.head.out_features
        
        # self.model.head.weight.requires_grad=False
        # self.model.head.bias.requires_grad=False
        parametrize.register_parametrization(  
            self.model.head, "weight", LoRAParametrization(features_in, features_out,3,1,self.device) )
        
    
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        # in this step update global model  (include base and global head) 
        for param in self.model.base.parameters():
            param.requires_grad = True

        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        
        

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            # self.learning_rate_scheduler_ghead.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, global_head):
        self.model.head.weight.data=global_head.weight.data.clone()
        self.model.head.bias.data=global_head.bias.data.clone()

