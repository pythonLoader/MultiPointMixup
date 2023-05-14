#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emd_ import emd_module
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class SageMix:
    def __init__(self, args, num_class=40):
        self.num_class = num_class
        
        # if self.kermix:
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        self.beta = torch.distributions.beta.Beta(torch.tensor([args.beta]), torch.tensor([args.beta]))

    
    def mixup(self, xyz, label, x=None, saliency=None):
        """
        Args:
            xyz (B,N,3): 
            label (B): 
            x (B,D,N): Defaults to None.
            saliency (B,N): Defaults to None.
        """        
        
        B, N, _ = xyz.shape
        idxs = torch.randperm(B)

        perm = xyz[idxs]
        
        _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
        ass = ass.long()
        perm_new = torch.zeros_like(perm).cuda()
        
        if saliency is not None:
            saliency_perm = torch.zeros_like(saliency).cuda()
        
        for i in range(B):
            perm_new[i] = perm[i][ass[i]]
            saliency_perm[i] = saliency[idxs][i][ass[i]]

        saliency = saliency/saliency.sum(-1, keepdim=True)
        
        anc_idx = torch.multinomial(saliency, 1, replacement=True)
        anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]
        
        sub = perm_new - anchor_ori[:,None,:]
        dist = ((sub) ** 2).sum(2).sqrt()
        
        saliency_perm = saliency_perm * dist
        saliency_perm = saliency_perm/saliency_perm.sum(-1, keepdim=True)
        
        anc_idx2 = torch.multinomial(saliency_perm, 1, replacement=True)
        anchor_perm = perm_new[torch.arange(B),anc_idx2[:,0]]
                
        alpha = self.beta.sample((B,)).cuda()

        sub_ori = xyz - anchor_ori[:,None,:]
        sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
        
        sub_perm = perm_new - anchor_perm[:,None,:]
        sub_perm = ((sub_perm) ** 2).sum(2).sqrt()   
        ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)
        
        weight_ori = ker_weight_ori * alpha 
        weight_perm = ker_weight_perm * (1-alpha)
        
        weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        weight = weight/weight.sum(-1)[...,None]

        x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        x = x.permute(0,2,1)
        
        #label generation
        target = weight.sum(1)
        target = target / target.sum(-1, keepdim=True)
        label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_perm_onehot = label_onehot[idxs]
        label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot 
        
        return x, label, {"mix" : x, "perm_idxs" : idxs,
                  "perm" : perm_new, "ker_weight_perm" : ker_weight_perm, "weight_perm":weight_perm,
                  "ker_weight_ori" : ker_weight_ori,"weight_ori":weight_ori,\
                  "saliency" : saliency, "saliency_perm" : saliency_perm, "ratio":weight}
    
    
    
    def base_mixup(self, xyz, label):
        batch_size = xyz.size(0)
        idxs = torch.randperm(batch_size)
        perm = xyz[idxs]
        _, ass = self.EMD(xyz, perm, 0.005, 300) # mapping
        ass = ass.long()
        perm_new = torch.zeros_like(perm).cuda()
        
        for i in range(batch_size):
            perm_new[i] = perm[i][ass[i]]
        
        beta = torch.distributions.beta.Beta(torch.tensor([0.4]), torch.tensor([0.4]))
        alpha = beta.sample((batch_size,1)).cuda()

        x = alpha * xyz + (1-alpha) * perm_new
        x = x.permute(0,2,1)
        
        alpha = alpha.squeeze(-1)
        label_onehot = torch.zeros(batch_size, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_perm_onehot = label_onehot[idxs]
        label = alpha* label_onehot + (1-alpha) * label_perm_onehot 
        
        return x, label
    


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)
        
        self.kermix=args.kermix
        self.basemix = args.basemix
        
        if self.kermix:
            self.SageMix = SageMix(args, output_channels)

    def forward(self, x, label=None, saliency=None):
        
        xyz = x.permute(0,2,1)
        
        #input mix
        if self.basemix and self.training:
            x, label = self.SageMix.base_mixup(xyz, label)
        elif self.kermix and self.training:
           x, label, temp = self.SageMix.mixup(xyz, label, saliency=saliency)
        
 
        # if mixup:
        #    
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x, label


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.num_class = output_channels
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.convs = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])
        
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        
        self.kermix=args.kermix
        self.basemix = args.basemix

        if self.kermix:
            self.SageMix = SageMix(args, output_channels)

    def forward(self, x, label=None, saliency=None, get_mix=False):
        batch_size = x.size(0)
        
        xyz = x.permute(0,2,1)
        xs=[]
        # input mix
        if self.basemix and self.training:
            x, label = self.SageMix.base_mixup(xyz, label)
        elif self.kermix and self.training:
           x, label, temp = self.SageMix.mixup(xyz, label, saliency=saliency)

        for i, conv in enumerate(self.convs):
            x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x = conv(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            xs.append(x)
            
        x = torch.cat(xs, dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)

        
        if get_mix:
            return x, label, temp
        else:
            return x, label
    
class Pointnet2_MSG(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pointnet2_MSG, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_channels)
        
        self.num_class=output_channels
        
        self.kermix=args.kermix
        self.basemix=args.basemix
        if self.kermix:
            self.SageMix = SageMix(args, output_channels)


    def forward(self, x, label=None, saliency=None):
        #xyz : b,3,n
        batch_size = x.size(0)
            
        xyz = x.permute(0,2,1)
        
        #input mix
        if self.basemix and self.training:
            x, label = self.SageMix.base_mixup(xyz, label)
        elif self.kermix and self.training:
           x, label, temp = self.SageMix.mixup(xyz, label, saliency=saliency)
        
        
        xyz = x
            
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(batch_size, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, label
