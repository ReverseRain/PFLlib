import torch
import torch.nn as nn
import time
from flcore.clients.clientbase import Client
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics

class LadderSideModule(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LadderSideModule, self).__init__()
        self.downsample = nn.Linear(feature_dim, feature_dim // 2)
        self.upsample = nn.Linear(feature_dim // 2, feature_dim)
        self.side_network = nn.Sequential(
            nn.Linear(feature_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, main_output, target_logits):
        # Downsample
        down = self.downsample(x)
        # Side network processing
        side_output = self.side_network(down)
        # Upsample
        up = self.upsample(down)
        return up, side_output

def initialize_weights_with_backbone(ladder_module, backbone_model):
    """
    Initialize the ladder module using the backbone's weights.
    """
    with torch.no_grad():
        # Initialize downsample layer
        if hasattr(backbone_model, 'fc1'):
            backbone_weights = backbone_model.fc1[0].weight.data  # Assuming fc1 is Sequential
            prune_ratio = ladder_module.downsample.out_features / backbone_weights.shape[0]
            pruned_weights = prune_weights(backbone_weights, prune_ratio)
            ladder_module.downsample.weight.copy_(pruned_weights)

        # Initialize upsample layer (transpose of downsample for reconstruction)
        if hasattr(ladder_module, 'downsample') and hasattr(ladder_module, 'upsample'):
            ladder_module.upsample.weight.copy_(ladder_module.downsample.weight.T)

def prune_weights(weight_matrix, prune_ratio):
    """
    Prune the weight matrix to retain top rows/columns based on importance.
    """
    num_features = int(weight_matrix.size(0) * prune_ratio)
    sorted_indices = torch.argsort(torch.abs(weight_matrix).sum(dim=1), descending=True)
    pruned_weights = weight_matrix[sorted_indices[:num_features], :]
    return pruned_weights


class clientDBE_LST(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.klw = args.kl_weight
        self.momentum = args.momentum
        self.global_mean = None
        self.ladder_module = LadderSideModule(feature_dim=512, num_classes=args.num_classes)

        self.client_mean = nn.Parameter(torch.zeros(512, device=self.device))
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)
        self.reset_running_stats()

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.ladder_module.to(self.device)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        # First phase: Train without DBE and LST to collect client-specific mean
        if self.global_mean is None:  # Only do this if not yet trained with DBE and LST
            self.collect_client_mean(trainloader)
            
        # Initialize LST weights
        if self.global_mean is not None and self.ladder_module.downsample.weight.sum() == 0:
            self.initialize_dbe_ladder(self.model)

        # Phase 2: Start training with DBE and LST
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                # ====== Feature extraction and ladder-side processing ======
                rep = self.model.base(x).to(self.device)
                running_mean = torch.mean(rep, dim=0)

                if self.global_mean is not None:
                    self.global_mean = self.global_mean.to(self.device)
                    up, side_output = self.ladder_module(rep, self.model.head(rep), self.global_mean)
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2)
                    output = self.model.head(rep + up + self.client_mean)
                    loss = self.loss(output, y) + reg_loss * self.klw
                else:
                    output = self.model.head(rep + self.client_mean)
                    loss = self.loss(output, y)

                # ====== Optimization ======
                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.opt_client_mean.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_client_mean(self, trainloader):
        """
        Train the model for one epoch to collect client-specific mean.
        """
        self.model.train()
        for x, y in trainloader:
            x, y = x.to(self.device), y.to(self.device)

            # Feature extraction to get the client-specific mean
            rep = self.model.base(x).to(self.device)
            running_mean = torch.mean(rep, dim=0)
            self.running_mean = running_mean.detach()

        # After training one epoch, update the global mean
        self.global_mean = self.running_mean

        # Freeze the original model's head after initializing LST and DBE
        for param in self.model.head.parameters():
            param.requires_grad = False

    def reset_running_stats(self):
        self.running_mean = torch.zeros_like(self.client_mean)
        self.running_mean.detach_()

    def detach_running(self):
        self.running_mean.detach_

    def initialize_dbe_ladder(self, backbone_model):
        """
        Initialize DBE and LST layers after receiving the weights from the server.
        """
        initialize_weights_with_backbone(self.ladder_module, backbone_model)
    
    # def set_parameters(self, model):
    
    #     # Iterate over the provided model's parameters and the client's model parameters
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.clone()

    #     # Handle the LST parameters separately
    #     for new_param, old_param in zip(model.ladder_module.parameters(), self.ladder_module.parameters()):
    #         old_param.data = new_param.data.clone()

    
    def train_metrics(self):
        trainloader = self.load_train_data()
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
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        reps = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)

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
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc