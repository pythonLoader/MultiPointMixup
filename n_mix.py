#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
import random

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


# from SageMix import SageMix
from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN
from util import cal_loss, cal_loss_mix, IOStream
import wandb
import torch.nn.functional as F
# import io

import torch
from emd_ import emd_module
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy

seed=1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class SageMix:
    def __init__(self, args, num_class=40):
        self.num_class = num_class
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        self.beta = torch.distributions.beta.Beta(torch.tensor([args.theta]), torch.tensor([args.theta]))

    # def find_optimal_mapping(self, xyz, saliency=None, theta=0.2):
    #     # print("saliency", saliency)
    #     B, N, _ = xyz.shape
    #     # print("xyz", xyz.shape)

    #     pairwise_emds = torch.zeros(B, B)
    #     pairwise_saliency_dist = torch.zeros(B, B)
    #     alignments = torch.zeros(B, B, N)
    #     for i in range(B):
    #         xyz_i = xyz[i].repeat(B, 1, 1)
    #         # print("xyz_i", xyz_i)
    #         emd_pointwise, ass = self.EMD(xyz_i, xyz, 0.005, 500)
    #         alignments[i] = ass
    #         # print("assignment", ass)
    #         emd_avg = emd_pointwise.sum(1)
    #         pairwise_emds[i, :] = emd_avg

    #     if saliency_based:
    #         for j in range(B):
    #             # print("alignments 0", alignments[0])
    #             saliency_perm = torch.zeros(B, N)
    #             for i in range(B):
    #                 # print("alignments", alignments[i, j].to(torch.int64))
    #                 saliency_perm[i] = saliency[i][alignments[i, j].to(torch.int64)]
    #             z = saliency_perm.unsqueeze(1)
                
    #             # print("z shape", z.shape)
    #             z_reshape = z.reshape(B, -1)
    #             # print("z_reshape:",z_reshape.shape)
    #             z_idx_1d = torch.argmax(z_reshape, dim=1)
    #             distances = z_idx_1d - z_idx_1d[j]
    #             pairwise_saliency_dist[j, :] = torch.abs(distances)
    #             # z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
    #             # z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
    #             # z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
    #             # print("z_idx_1d:",z_idx_1d)
    #             # A_dist = distance(z_idx_2d, dist_type='l1')
    #             # print("A_dist:", A_dist)

    #             # n_input = saliency.shape[0]
                
    #             # A_base = torch.eye(n_input, device=out.device)

    #             # A_dist = A_dist / torch.sum(A_dist) * n_input
    #             # m_omega = torch.distributions.beta.Beta(theta, theta).sample()
    #             # A = (1 - m_omega) * A_base + m_omega * A_dist
    #             # print("A", A)

    #     if not saliency_based: costs = pairwise_emds
    #     else: costs = pairwise_saliency_dist

    #     # for i in range(2):
    #     #     if i==0: costs = pairwise_emds
    #     #     if i==1: costs = pairwise_saliency_dist
    #     row_sums = torch.sum(costs, dim=1, keepdim=True)
        

    #     # Divide each row by its sum to normalize
    #     normalized_emds = costs / row_sums
    #     A_base = torch.eye(B)
    #     omega = torch.distributions.beta.Beta(theta, theta).sample()
    #     # print(omega)
    #     cost = omega * A_base + (1-omega) * normalized_emds
    #     # print("costs", cost)
        
    #     # print("normalized costs", A)
    #     row_ind, col_ind = linear_sum_assignment(cost)
    #     mapping = torch.zeros(B, 2)
    #     mapping[:, 0] = torch.tensor(row_ind)
    #     mapping[:, 1] = torch.tensor(col_ind)

    #     # To get the minimum cost
    #     # min_cost = cost[row_ind, col_ind].sum()

    #     # print("Assigned Indices:")
    #     # print(list(zip(row_ind, col_ind)))
    #     # print("pairwise saliency dist", pairwise_saliency_dist)
    #     return mapping

    def find_optimal_mapping(self, xyz, saliency=None, theta=0.2, k=3):
        # print("saliency", saliency)
        B, N, _ = xyz.shape
        # print("xyz", xyz.shape)

        pairwise_saliency_dist = torch.zeros(B, B)
        # print(saliency.shape)
        # max_saliencies = saliency.max(dim=1)
        max_saliencies = torch.topk(saliency, k=k, dim=1)
        # print("max_saliencies shape", max_saliencies.shape)
        # idxs = max_saliencies.indices
        idxs = torch.topk(saliency, k=k, dim=1).indices

        # Add an extra dimension to the arange tensor to match the dimensionality of 'idxs'.
        rows = torch.arange(xyz.size(0)).unsqueeze(-1)

        # Use advanced indexing to extract the desired elements.
        xyz_max = xyz[rows, idxs]

        # xyz_flat = xyz.view(xyz.shape[0], -1)

        # # Compute pairwise distances using broadcasting and vectorized operations.
        # diff = xyz_flat.unsqueeze(1) - xyz_flat.unsqueeze(0)
        # distances = torch.sqrt((diff ** 2).sum(-1))
        # print("distances", distances)

        # row_sums = distances.sum(dim=1, keepdim=True)

        # # Divide each row by its sum to normalize it.
        # normalized_distances = distances / row_sums
        # print("normalized distances", normalized_distances)
        # xyz = torch.randn(24, 5, 3)

        # Reshape xyz for broadcasting.
        xyz_reshaped = xyz_max.unsqueeze(1)

        # Compute differences between all pairs of points.
        diff = xyz_reshaped - xyz_reshaped.transpose(0, 1)

        # Compute squared Euclidean distances.
        sq_distances = (diff ** 2).sum(-1)

        # Compute Euclidean distances and take the mean over the second dimension.
        distances = sq_distances.sqrt().mean(2)

        row_sums = distances.sum(dim=1, keepdim=True)

        # Divide each row by its sum to normalize it.
        normalized_distances = distances / row_sums
        max_score = torch.max(distances)
        distances_for_minimization = max_score - normalized_distances

        max_score = torch.max(distances_for_minimization)

        row_sums = distances_for_minimization.sum(dim=1, keepdim=True)

        # Divide each row by its sum to normalize it.
        normalized_distances = distances_for_minimization / row_sums

        # print("distances", normalized_distances)

        
        A_base = torch.eye(B).to(device)
        omega = torch.distributions.beta.Beta(theta, theta).sample()
        # # print(omega)
        cost = omega * A_base + (1-omega) * normalized_distances
        # print("omega", omega)
        # print("costs", cost)
        
        # # print("normalized costs", A)
        mapping = torch.zeros(B, 10).to(device)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
        mapping[:, 0] = torch.tensor(row_ind)
        mapping[:, 1] = torch.tensor(col_ind)
        
        cost[mapping[:, 0].to(torch.int64), mapping[:, 1].to(torch.int64)] = 1000

        for i in range(2,5):
            # print("cost", cost)
            row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())
            mapping[:, i] = torch.tensor(col_ind)
            # print("mapping", mapping)
            cost[mapping[:, 0].to(torch.int64), mapping[:, i].to(torch.int64)] = 1000



        # print("mapping", mapping)

        # # To get the minimum cost
        # # min_cost = cost[row_ind, col_ind].sum()

        # # print("Assigned Indices:")
        # # print(list(zip(row_ind, col_ind)))
        # # print("pairwise saliency dist", pairwise_saliency_dist)
        return mapping.to(torch.int64)
    



    # def permute(self, xyz, label, saliency=None, n_clouds=2):

    def mix(self, xyz, label, saliency=None, n_mix=4, theta=0.2):
        """
        Args:
            xyz (B,N,3)
            label (B)
            saliency (B,N): Defaults to None.
        """        
        # label_ori = label.clone()
        # print(xyz.shape)
        B, N, _ = xyz.shape
        # print("saliency based", saliency_based)
        mapping = self.find_optimal_mapping(xyz, saliency)
        idxs = mapping.T[:n_mix, :]
        # print("idxs", idxs)
        
        # print(mapping)
        # return 0
        # print(xyz.shape)
        # idxs = mapping[:,1].to(torch.int64) #torch.randperm(B)
        # idxs = torch.stack([torch.randperm(B) for _ in range(n_mix-1)])

        # print("idxs", idxs)
        # return 0
        # idxs = torch.argsort(torch.rand(B, n_clouds))
        # saliency_sagemix = saliency.clone()

        xyzs = torch.zeros((n_mix, B, N, 3)).cuda()
        for i in range(n_mix):
            if i == 0: xyzs[i] = xyz
            else:
                xyzs[i] = xyz[idxs[i]]


        all_xyz = torch.zeros((n_mix, B, N, 3)).cuda()
        all_xyz[0] = xyzs[0]

        all_saliency = torch.zeros((n_mix, B, N)).cuda()
        all_saliency[0] = saliency
        for i in range(1, n_mix):
            # print("EMD inputs", xyzs[0][0], xyzs[i][0])
            _, ass = self.EMD(xyzs[0], xyzs[i], 0.005, 500)
            # print("assignment", ass)

            xyz_new = torch.zeros_like(xyzs[i]).cuda()
            saliency_new = torch.zeros_like(saliency).cuda()
            # print(idxs[i-1])
            
            # print(ass,ass.shape)
            for j in range(B):
                # print("ass j", ass[j])
                # print("ass j shape", ass[j].shape)
                # print("xyzs shape", xyzs.shape)
                # print("xyzs i shape", xyzs[i].shape)
                all_xyz[i][j] = xyzs[i][j][ass[j]]
                all_saliency[i][j] = saliency[idxs[i]][j][ass[j]]

                # xyz_new[i] = xyzs[j][ass[j]]
                # saliency_new[j] = saliency[idxs[j]][j][ass[j]]
            
            # all_xyz[i] = xyz_new
            # all_saliency[i] = saliency_new
        # print("permuted saliency", saliency[1])

        anchors = torch.zeros(n_mix, B, 3).cuda()

        saliency = saliency/saliency.sum(-1, keepdim=True)
        # anc_idx = torch.randint(0, 1024, (B,1)).cuda()
        anc_idx = torch.multinomial(saliency, 1, replacement=True)
        anchor_ori = all_xyz[0][torch.arange(B), anc_idx[:,0]]
        anchors[0] = anchor_ori
        # # print("anchor shape", anchor_ori.shape)
        # print("saliency", saliency)
        

        anc_idx_new = 0
        perm_saliency_new = 0
        # ker_weight_fix = 0
        for i in range(1, n_mix):
            dists = []
            for j in range(0,i):
                # print("all_xyz", all_xyz[i])
                # print("anchors", anchors)
                sub = all_xyz[i] - anchors[j][:, None, :]
                # subs.append(sub)
                dist = ((sub) ** 2).sum(2).sqrt()
                dists.append(dist)
                # print(dist.shape)
            dist = torch.stack(dists).sum(dim=0)
            
            perm_saliency_new = all_saliency[i] * dist
            perm_saliency_new = perm_saliency_new/perm_saliency_new.sum(-1, keepdim=True)


        #     ## try to fix this at 0
            anc_idx_new = torch.multinomial(perm_saliency_new, 1, replacement=True)
            # anc_idx_new = torch.randint(0, 1024, (B,1)).cuda()
            anchor_perm_new = all_xyz[i][torch.arange(B),anc_idx_new[:,0]]
            anchors[i] = anchor_perm_new
            # sub = perm_new - anchor_ori[:,None,:]
        #     # dist = ((sub) ** 2).sum(2).sqrt()
        #     # perm_saliency = perm_saliency * dist
        #     # perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
        # # alpha = self.dirichlet.sample((B,)).cuda()
        pi = torch.distributions.dirichlet.Dirichlet(torch.tensor([theta for i in range(n_mix)])).sample((B,)).cuda()
        # print("anc_idx_new", anc_idx_new)
        # print("perm saliency", perm_saliency_new)
        # alpha = self.beta.sample((B,)).cuda()
        # print("alpha", alpha)
        # print("idxs", idxs)
        # print("alpha nmix", alpha)
        # # print("pi shape", pi.shape)
        # # print("pi sum", pi.sum(1))
        

        kerns = torch.zeros(n_mix, B, N).cuda()
        weights = torch.zeros(n_mix, B, N).cuda()
        weights_copy = []
        for i in range(n_mix):
            sub_ori = all_xyz[i] - anchors[i][:,None,:]
            sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        #     #Eq.(6) for first sample
            ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
            kerns[i] = ker_weight_ori
        #     # print("kern weight ori", ker_weight_ori.shape)

            weights[i] = ker_weight_ori * pi[:,i][:,None]
            # if i == 0: weights[i] = ker_weight_ori * alpha
            # else: weights[i] = ker_weight_ori * (1-alpha) 
            weights_copy.append(weights[i][...,None])

            # ker_weight_fix = ker_weight_ori


        # # weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        weight = (torch.cat(weights_copy,-1)) + 1e-16
        weight = weight/weight.sum(-1)[...,None]

        # weight_old = weight.clone()
        x_nmix = torch.zeros((B, N, 3)).cuda()

        for i in range(n_mix):
            x_nmix += weight[:, :, i:i+1] * all_xyz[i]
        target = weight.sum(1)
        target = target / target.sum(-1, keepdim=True)

        label_one_hots = torch.zeros(n_mix, B, self.num_class).cuda()
        label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_one_hots[0] = label_onehot
        # print("label_onehot shape", label_onehot.shape)

        label_nmix = torch.zeros(B, self.num_class).cuda()
        label_nmix += label_one_hots[0] * target[:, 0, None]
        
        for i in range(1, n_mix):
            label_perm_onehot = label_onehot[idxs[i]]
            label_nmix += label_perm_onehot * target[:, i, None]
        # label
        # label_old = label
        # x_old = x

        # idxs = torch.randperm(B)
        # idxs = idxs[0]
        # # print(idxs)

        
        # #Optimal assignment in Eq.(3)
        # perm = xyz[idxs]
        
        
        # # _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
        # # ass = ass.long()
        # perm_new = torch.zeros_like(perm).cuda()
        # perm_saliency = torch.zeros_like(saliency).cuda()
        
        # # print(ass,ass.shape)
        # for i in range(B):
        #     perm_new[i] = perm[i][ass[i]]
        #     perm_saliency[i] = saliency_sagemix[idxs][i][ass[i]]

        # # print("perm new diff", torch.sum(torch.abs(perm_new - all_xyz[1])))
        # # print("perm saliency diff", torch.sum(torch.abs(perm_saliency - all_saliency[1])))
        
        # #####
        # # Saliency-guided sequential sampling
        # #####
        # #Eq.(4) in the main paper
        # saliency_sagemix = saliency_sagemix/saliency_sagemix.sum(-1, keepdim=True)


        # # anc_idx = anc_idx_new #torch.multinomial(saliency_sagemix, 1, replacement=True)
        # anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]
        
        # #cal distance and reweighting saliency map for Eq.(5) in the main paper
        # sub = perm_new - anchor_ori[:,None,:]
        # dist = ((sub) ** 2).sum(2).sqrt()
        # perm_saliency = perm_saliency * dist
        # perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
        
        # #Eq.(5) in the main paper
        # anc_idx2 = anc_idx_new # torch.multinomial(perm_saliency, 1, replacement=True)
        # anchor_perm = perm_new[torch.arange(B),anc_idx2[:,0]]
                
                
        # #####
        # # Shape-preserving continuous Mixup
        # #####
        # # alpha = self.beta.sample((B,)).cuda()
        # sub_ori = xyz - anchor_ori[:,None,:]
        # sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        # #Eq.(6) for first sample
        # ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
        
        # sub_perm = perm_new - anchor_perm[:,None,:]
        # sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
        # #Eq.(6) for second sample
        # ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)
        
        # # print("alpha sagemix", alpha)
        # #Eq.(9)
        # weight_ori = ker_weight_ori * alpha 
        # weight_perm = ker_weight_perm * (1-alpha)
        # weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        # weight = weight/weight.sum(-1)[...,None]

        # #Eq.(8) for new sample
        # x_sagemix = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        
        # #Eq.(8) for new label
        # target = weight.sum(1)
        # target = target / target.sum(-1, keepdim=True)
        
        # # if mixing_idx == 0:
        # label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        # label_perm_onehot = label_onehot[idxs]
        # label_sagemix = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot

        # # else:
        # #     label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        

        # # label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot
    
        # # x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        # # print("weight shape", weight.shape)

            
        # # weight_ori = ker_weight_ori * alpha 
        # # weight_perm = ker_weight_perm * (1-alpha)
        # # weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        # # weight = weight/weight.sum(-1)[...,None]
        
        # # print("all xyz", all_xyz[1])
        # # print("all saliency", all_saliency[1])

        # # print
        # # print(perms)

        # # assignments = torch.zeros((n_clouds, B, N))

        # # for i in range(n_clouds):


        # # idxs = torch.randperm(B)
        # # idxs = idxs[1]
        # # # # Optimal assignment in Eq.(3)
        # # perm = xyz[idxs]
        
        # # # _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
        # # # ass = ass.long()
        # # perm_new = torch.zeros_like(perm).cuda()
        # # perm_saliency = torch.zeros_like(all_saliency[0]).cuda()
        # # saliency = all_saliency[0]
        
        # # # # print(ass,ass.shape)
        # # for i in range(B):
        # #     perm_new[i] = perm[i][ass[i]]
        # #     perm_saliency[i] = saliency[idxs][i][ass[i]]
        
        # # # print("perm new diff", torch.sum(torch.abs(perm_new - all_xyz[1])))
        # # # print("saliency diff", torch.sum(torch.abs(perm_saliency - all_saliency[1])))
        # # # print("perm saliency", perm_saliency)
        # # # print("all saliency", all_saliency[1])
        
        # # # print("perm new", perm_new)
        # # # print("xyzs[1]", all_xyz[1])
        # # # print("perm new diff", torch.sum(torch.abs(perm_new - all_xyz[1])))
        # # # print("perm saliency diff", torch.sum(torch.abs(perm_saliency - all_saliency[1])))
        # # # # return 0
        # # # #####
        # # # # Saliency-guided sequential sampling
        # # # #####
        # # # #Eq.(4) in the main paper
        # # saliency = saliency/saliency.sum(-1, keepdim=True)
        # # # anc_idx = torch.multinomial(saliency, 1, replacement=True)
        # # anc_idx = torch.randint(0, 1024, (B,1)).cuda()
        # # print(anc_idx)
        # anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]
        
        # # #cal distance and reweighting saliency map for Eq.(5) in the main paper
        # sub = perm_new - anchor_ori[:,None,:]
        # dist = ((sub) ** 2).sum(2).sqrt()
        # perm_saliency = perm_saliency * dist
        # perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
        # # print("perm saleincy new diff", torch.sum(torch.abs(perm_saliency_new - perm_saliency)))

        
        
        # # #Eq.(5) in the main paper
        # # anc_idx2 = torch.multinomial(perm_saliency, 1, replacement=True)
        # anc_idx2 = anc_idx_new
        # anchor_perm = perm_new[torch.arange(B), anc_idx2[:,0]]
                
                
        # # #####
        # # # Shape-preserving continuous Mixup
        # # #####
        # # alpha = self.beta.sample((B,)).cuda()
        # # # print("2-class pi shape", alpha.shape)
        # sub_ori = xyz - anchor_ori[:,None,:]
        # sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        # # #Eq.(6) for first sample
        # ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
        # print("kernel ori weight diff", torch.sum(torch.abs(ker_weight_ori - kerns[0])))
        # # # print("kernel shape", ker_weight_ori.shape)
        
        # sub_perm = perm_new - anchor_perm[:,None,:]
        # sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
        # # #Eq.(6) for second sample
        # ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)
        # print("kernel perm weight diff", torch.sum(torch.abs(ker_weight_perm - kerns[1])))
        
        # # #Eq.(9)
        # alpha = pi[:,0][:,None]
        # weight_ori = ker_weight_ori * alpha 
        # # # print("weight shape", weight_ori.shape)
        # weight_perm = ker_weight_perm * (1-alpha)

        # # print(torch.sum(torch.abs((1-alpha) - pi[:,1][:,None])))
        # # print(pi[:,1][:,None] +pi[:,0][:, None] )
        # # print( pi[:,1][:,None] + alpha)
        # # print((1-alpha) + pi[:,0][:, None])

        # # print("weight ori diff", torch.sum(torch.abs(weight_ori - weights[0])))
        # # print("weight perm diff", torch.sum(torch.abs(weight_perm - weights[1])))



        # weight_new = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
        # weight_new = weight/weight.sum(-1)[...,None]
        # # # print("weight shape", weight.shape)
        # # print("weight 0 diff", torch.sum(torch.abs(weight_ori - weights[0])))
        # # print("perm diff", torch.sum(torch.abs(perm_new - all_xyz[1])))
        
        # # print("weight 1 diff", torch.sum(torch.abs(weight_perm - weights[1])))
        # # print("weight diff", torch.sum(torch.abs(weight_new - weight_old)))

        # # #Eq.(8) for new sample
        # x_new = weight_new[:,:,0:1] * xyz + weight_new[:,:,1:] * perm_new
        # # # print("x shape", x.shape)
        
        
        # # #Eq.(8) for new label
        # target = weight_new.sum(1)
        # target = target / target.sum(-1, keepdim=True)
        
        
        # label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label_ori.view(-1, 1), 1)
        # label_perm_onehot = label_onehot[idxs]
        # label_new = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot
        # print("label diff", torch.sum(torch.abs(label_sagemix - label_nmix)))
        # print("x diff", torch.sum(torch.abs(x_sagemix - x_nmix)))
        # print("label", label_nmix)
        # print("x", x_nmix)
        return x_nmix, label_nmix
    

def distance(z, dist_type='l2'):
    '''Return distance matrix between vectors'''
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist


def calc_A_dist(saliency, theta=0.5):
    sc = saliency.unsqueeze(1)
    # print("sc:",sc.shape)
    # z = F.avg_pool1d(sc, kernel_size=8, stride=1)
    # print("z:",z.shape)
    z = sc
    z_reshape = z.reshape(args.batch_size, -1)
    # print("z_reshape:",z_reshape.shape)
    z_idx_1d = torch.argmax(z_reshape, dim=1)
    z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
    z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
    z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
    # print("z_idx_2d:",z_idx_2d)
    A_dist = distance(z_idx_2d, dist_type='l1')
    # print("A_dist:", A_dist)

    n_input = saliency.shape[0]
    
    A_base = torch.eye(n_input, device=out.device)

    A_dist = A_dist / torch.sum(A_dist) * n_input
    m_omega = torch.distributions.beta.Beta(theta, theta).sample()
    A = (1 - m_omega) * A_base + m_omega * A_dist
    # print("A", A)
    return A



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    if args.data == 'MN40':
        dataset = ModelNet40(partition='train', num_points=args.num_points)
        # args.batch_size = len(dataset)
        args.batch_size = 24
        print('args.batch_size:',args.batch_size)
        train_loader = DataLoader(dataset, num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class=40
    elif args.data == 'SONN_easy':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="easy"), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="easy"), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    elif args.data == 'SONN_hard':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="hard"), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="hard"), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    
    
    device = torch.device("cuda" if args.cuda else "cpu")

    print(args.model)
    #Try to load models
    if args.model == 'pointnet':
        print('pointnet!!')
        model = PointNet(args, num_class).to(device)
    elif args.model == 'dgcnn':
        print('dgcnn!!')
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    # modelA = ModelA()
    # modelB = ModelB()
    # state_dict = model.state_dict()
    # new_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
    # model.load_state_dict(new_state_dict)

    print(args.fine_tune)
    if args.fine_tune:
        # Load the original state_dict (with 'module.' prefix)
        state_dict = torch.load(args.fine_tune)

        # Create a new state_dict without 'module.' prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the new state_dict into model
        model.load_state_dict(new_state_dict)

        # model.load_state_dict(torch.load(args.fine_tune))

    # print("conv1 weights:", model.conv1[0].weight.data)
    # print("linear1 weights:", model.linear1.weight.data) 

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")


    if args.use_sgd:
        print("Use SGD")
        if not args.fine_tune:
            opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print(type(args.lr), args.lr)
            print(type(args.last_epoch), args.last_epoch)
            lr = 0 + 0.5 * (0.1 - args.lr) * (1 + np.cos(np.pi * args.last_epoch / 500))
            opt = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        if not args.fine_tune:
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        else:
            # print()
            lr = 0 + 0.5 * (0.1 - args.lr) * (1 + np.cos(np.pi * args.last_epoch / 500))
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    

    sagemix = SageMix(args, num_class)
    criterion = cal_loss_mix
    # print(args)

    mixup = "random" if args.fixed_mixup is None else "fixed {}".format(args.fixed_mixup)

    wandb_name = "finetuning-" + args.model + "-" + args.data
    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_name,
        
        # track hyperparameters and run metadata
        config={
        # "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.data,
        "last epoch": args.last_epoch,
        "classes": args.fixed_mixup,
        "implementation": "nmix",
        }
    )

    # for layer in model.children():
    #     print("layer:", layer)
    #     if isinstance(layer, nn.Linear):
    #         print(layer.state_dict()['weight'])
    #         print(layer.state_dict()['bias'])
    # print("linear1 weights:", model.module.linear1.weight.data)
    # torch.save(model.state_dict(), 'nmix_weights.pth')
       

    best_test_acc = 0
    for epoch in range(args.last_epoch, args.epochs):

        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            # print("data shape", data.shape)
            batch_size = data.size()[0]
            
            ####################
            # generate augmented sample
            ####################
            if args.fixed_mixup is None: 
                n_mix = random.randint(1,4)
            else: 
                n_mix = int(args.fixed_mixup)

            
            
            # print("n_mix", n_mix)
            if n_mix > 1:
                model.eval()
                data_var = Variable(data.permute(0,2,1), requires_grad=True)
                logits = model(data_var)
                loss = cal_loss(logits, label, smoothing=False)
                # print("eval loss", loss)
                loss.backward()
                opt.zero_grad()
                saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
                
                # saliency_based=False
                # if args.mapping is not 'emd': saliency_based=True
                # print("saliency", saliency)
                data, label = sagemix.mix(data, label, saliency, n_mix)
                # print("data, label", data, label)
                # print("data, label", data[0], label[0])

            
                # mixed_saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
            # print("data shape", data.shape)
            # break
            
            model.train()
            # break
                
            opt.zero_grad()
            # print("data, label", data.permute(0,2,1)[0], label[0])
            logits = model(data.permute(0,2,1))
            if n_mix > 1:
                loss = criterion(logits, label)
            else:
                loss = cal_loss(logits, label)
           
            # print("logits", logits[0])
            # print("loss", loss)
            # exit()
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
        
        # if epoch % 10 == 0:
        #     print("label", torch.topk(label, k=3, dim=1).values)
        
        scheduler.step()
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = cal_loss(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(model.state_dict(), 'finetuned_models/FINETUNED_{}mix_dataset_{}_model_{}_epochs_{}.pth'.format(args.fixed_mixup, args.data, args.model, epoch))
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              best_test_acc)

        
        
        wandb.log({"Test acc": test_acc, "test avg acc": avg_per_class_acc, "best test acc": best_test_acc, "epoch": epoch})
        wandb.log({"n_mix": n_mix})
        io.cprint(outstr)

    wandb.finish()
       


def test(args, io):
    if args.data == 'MN40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class=40
    elif args.data == 'SONN_easy':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="easy"), 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    elif args.data == 'SONN_hard':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="hard"), 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    device = torch.device("cuda" if args.cuda else "cpu")

    

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, num_class).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

    


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--data', type=str, default='MN40', metavar='N',
                        choices=['MN40', 'SONN_easy', 'SONN_hard']) #SONN_easy : OBJ_ONLY, SONN_hard : PB_T50_RS
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--m_omega', type=int, default=0.9,
                        help='omega parameter')
    parser.add_argument('--mapping', type=str, default='emd',
                        help='mapping function')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--fixed_mixup', type=str, default=None, metavar='N',
                        help='number of mixes')
    parser.add_argument('--last_epoch', type=str, default=0)
    parser.add_argument('--fine_tune', type=str, default=False)
                        
    
    
    
    parser.add_argument('--sigma', type=float, default=-1) 
    parser.add_argument('--theta', type=float, default=0.2) 
    args = parser.parse_args()

    print(args)
    if args.fine_tune:
        str_split = args.fine_tune.split("_")
        print(str_split)
        args.model = str_split[-3]
        args.last_epoch = int(str_split[-1].split(".")[0])
    # args.data = str_split[2] + "_" + str_split[3]


    if args.sigma==-1:
        if args.model=='dgcnn':
            args.sigma=0.3
        elif args.model=="pointnet":
            args.sigma=2.0
    
    if args.model=='dgcnn':
        args.use_sgd=True
    elif args.model=="pointnet":
        args.use_sgd=False

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)





