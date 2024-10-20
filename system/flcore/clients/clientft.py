import copy
import torch.nn as nn
import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize 
from flcore.clients.clientbase import Client

class FourierParametrization(nn.Module):  
    def __init__(self, features_in, features_out, n=3, alpha=1, device='cpu'):  
        super().__init__()  
        
        self.device=device
        self.features_in=features_in
        self.features_out=features_out
        #entry initialization
        self.E=torch.randperm(features_in*features_out)[:n]
        self.E=torch.stack([self.E//self.features_in,self.E%self.features_out],dim=0).to(device)

        #spectral coefficient initialization
        # 这里的c就不放到cuda上面了,因为之后还要根据这个生成稀疏矩阵等，都会消耗cuda的内存
        self.c= nn.Parameter(torch.randn(n).to(self.device),requires_grad=True)

        self.scale = alpha / n  # 缩放因子  
  
    def forward(self, original_weights): 
        # 获得变量代表的稀疏矩阵
        F=torch.zeros(self.features_out,self.features_in).to(self.device)
        F[self.E[0,:],self.E[1,:]]=self.c
        # 通过对稀疏矩阵进行逆傅里叶变化 获得W
        w=(torch.fft.ifft2(F).real*self.scale).to(self.device)
        
        # 不能只有w.to(self.device)还要就地赋值改变。
        return original_weights + w

class clientFT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda

        features_in_fc=self.model.head.fc.in_features
        features_out_fc=self.model.head.fc.out_features
        
        
        parametrize.register_parametrization(  
            self.model.head.fc, "weight", FourierParametrization(features_in_fc, features_out_fc,features_in_fc//3,1,self.device) )
        
        features_in_fc1=self.model.head.fc1[0].in_features
        features_out_fc1=self.model.head.fc1[0].out_features
        
        parametrize.register_parametrization(  
            self.model.head.fc1[0], "weight", FourierParametrization(features_in_fc1, features_out_fc1,features_in_fc1//3,2,self.device) )
    
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

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, global_head):
        self.model.head.fc.weight.data=global_head.fc.weight.data.clone()
        self.model.head.fc.bias.data=global_head.fc.bias.data.clone()
        self.model.head.fc1[0].weight.data=global_head.fc1[0].weight.data.clone()
        self.model.head.fc1[0].bias.data=global_head.fc1[0].bias.data.clone()
        
        