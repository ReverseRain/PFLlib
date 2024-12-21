import copy
import torch.nn as nn
import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize 
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client

class LoRAadpater(nn.Module):  
    def __init__(self, features_in, features_out, rank=3, alpha=1, device='cpu'):  
        super().__init__()  
        # 论文第4.1节：  
        # 我们使用随机高斯初始化A，并将B初始化为零，所以ΔW = BA在训练开始时为零  
        self.lora_A = nn.Parameter(torch.zeros((rank, features_in)).to(device))  # 初始化为零的低秩矩阵A  
        self.lora_B = nn.Parameter(torch.zeros((features_out, rank)).to(device))  # 初始化为零的低秩矩阵B 
        
        nn.init.normal_(self.lora_A, mean=0, std=1)  # 对A进行标准正态分布初始化  
  
        # self.scale = alpha / rank  # 缩放因子  
  
    def forward(self, x): 
        weight=torch.matmul(self.lora_B, self.lora_A)
        return   torch.matmul(x,weight.T.to(torch.float))
    


class clientLoRA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda

        features_in=self.model.head.in_features
        features_out=self.model.head.out_features
        
        # self.model.head.weight.requires_grad=False
        # self.model.head.bias.requires_grad=False
        self.adpater=LoRAadpater(features_in, features_out,3,1,self.device)

        self.opt_adpater = torch.optim.SGD(self.adpater.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_adpater = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt_adpater, 
            gamma=args.learning_rate_decay_gamma
        )

        
    
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        u=0.3
        # 第一次训练的过程中冻结adapter 而训练模型    
        for param in self.adpater.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
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
                tem = self.model.base(x)
                output1=self.adpater(tem)
                loss = self.loss(output1, y)

                output2=self.model(x)

                loss=u*self.loss(output1,y)+(1-u)*self.loss(output2,y)
                
                self.opt_adpater.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.opt_adpater.step()
                self.optimizer.step()

        # 第二次训练的过程中冻结模型  而训练adapter  
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.adpater.parameters():
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
                tem = self.model.base(x)
                output=self.adpater(tem)
                loss = self.loss(output, y)

                self.opt_adpater.zero_grad()
                loss.backward()
                self.opt_adpater.step()
        
        
        

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_adpater.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, global_adpater):
        for new_param, old_param in zip(global_adpater.parameters(), self.adpater.parameters()):
            old_param.data = new_param.data.clone()
        
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                tem=self.model.base(x)
                # print(tem.shape,self.adpater)
                output = self.model.head(tem)+0.3*self.adpater(tem)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                tem=self.model.base(x)
                output = self.model.head(tem)+0.3*self.adpater(tem)

                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
        
