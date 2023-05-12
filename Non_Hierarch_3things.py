from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN
from util import cal_loss, cal_loss_mix, IOStream
import gco
import os

# Specify which GPUs to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = argparse.Namespace(batch_size=40, data='MN40', dropout=0.5, emb_dims=1024, epochs=100, eval=False, exp_name='SageMix', k=20, lr=0.0001, model='pointnet', model_path='', momentum=0.9, no_cuda=False, num_points=1024, seed=1, sigma=-1, test_batch_size=16, theta=0.2, use_sgd=False)

num_points = 1024
dataset = ModelNet40(partition='train', num_points=num_points)
batch_size=args.batch_size

test_batch_size = args.test_batch_size
train_loader = DataLoader(dataset, num_workers=8,
                        batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points), num_workers=8,
                        batch_size=test_batch_size, shuffle=True, drop_last=False)
num_class=40

if args.data == 'MN40':
    dataset = ModelNet40(partition='train', num_points=args.num_points)
    # args.batch_size = len(dataset)
    # args.batch_size = 40
    #print('args.batch_size:',args.batch_size)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Try to load models
if args.model == 'pointnet':
    model = PointNet(args, num_class).to(device)
elif args.model == 'dgcnn':
    model = DGCNN(args, num_class).to(device)
else:
    raise Exception("Not implemented")

import torch
from emd_ import emd_module
import time
torch.manual_seed(0)
class SageMix:
    def __init__(self, args, device, num_class=40):
    
        self.num_class = num_class
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        # self.beta = torch.distributions.beta.Beta(torch.tensor([args.theta]), torch.tensor([args.theta]))
        self.dirich = torch.distributions.dirichlet.Dirichlet(torch.tensor([args.theta,args.theta,args.theta]))
        self.device = device
    
    
    def find_alignment_and_mapping(self,row):
        MB = row.shape[0]
        

       # Expand row into a (MB, MB, 1024, 3) tensor
        row_expanded = row.unsqueeze(1).expand(-1, MB, -1, -1)
        perm2 = row_expanded.clone()

        row_repeated = torch.repeat_interleave(row.unsqueeze(0), repeats=MB, dim=0)
        
        # Assuming self.EMD() can handle batches
        dist2, ass2 = self.EMD(row_repeated.view(-1,1024,3),perm2.view(-1,1024,3), 0.005, 500) # mapping
        
        # Reshape dist and ass back to (MB, MB, 1024)
        dist_mat = dist2.view(MB, MB, 1024)
        ass_mat = ass2.view(MB, MB, 1024)
        dist_mat = torch.norm(dist_mat,dim=2)
        avg_alignment_dist = torch.mean(dist_mat,dim=0)
        min_idx = torch.argmin(avg_alignment_dist).item()

        return min_idx,ass_mat[min_idx]

    def derangement(self,size):
        while True:
            perm = torch.randperm(size)
            if (perm != torch.arange(size)).all():
                return perm
    

    def mix(self, xyz, label, saliency=None, mixing_idx=0):
        """
        Args:
            xyz (B,N,3)
            label (B)
            saliency (B,N): Defaults to None.
        """        
        # print(xyz.shape)
        B, N, _ = xyz.shape
        
        # print(xyz.shape)
        idxs1 = self.derangement(B)
        idxs2 = self.derangement(B)
        while True:
            if (idxs1 != idxs2).all():
                break
            else:
                idxs2 = self.derangement(B)
        
        perm1 = xyz[idxs1]
        perm2 = xyz[idxs2]
        optimal_pcd_idx = torch.empty(B)
        # assignment = []
        batch_pcd = torch.stack((xyz,perm1,perm2))
    
        # print(batch_pcd.shape)
        batch_pcd = batch_pcd.permute(1,0,2,3)
        # print("batch_pcd:",batch_pcd.shape)
        # self.find_alignment_and_mapping(row) for row in batch_pcd
        # start = time.time()
        ret = [self.find_alignment_and_mapping(row) for row in batch_pcd] # [3,1024,3]
        
        # print('time for finding optimal alignment: ', time.time() - start)
        # return 
        # print(ret)
        ret_1, ret_2 = zip(*ret)
        optimal_pcd_idx = torch.tensor(ret_1)

        assignment = torch.stack(ret_2)

        # print("index array:",optimal_pcd_idx.shape)
        # print("assignment:",assignment.shape)
        # return
        #Optimal assignment in Eq.(3)
        
        assignment = assignment.long()
        xyz_new = torch.zeros_like(xyz).cuda()
        xyz_saliency = torch.zeros_like(saliency).cuda()

        perm1_new = torch.zeros_like(perm1).cuda()
        perm1_saliency = torch.zeros_like(saliency).cuda()

        perm2_new = torch.zeros_like(perm2).cuda()
        perm2_saliency = torch.zeros_like(saliency).cuda()

        # print(assignment[:,0,:].dtype,assignment[:,0,:].shape)
        # print("saliency mat:",saliency.shape)

        for i in range(B):
            xyz_new[i] = xyz[i][assignment[i,0]]
            # print(saliency[i].shape,assignment[i,0].shape)
            xyz_saliency[i] = saliency[i][assignment[i,0]]
            perm1_new[i] = perm1[i][assignment[i,1]]
            perm1_saliency[i] = saliency[i][assignment[i,1]]
            perm2_new[i] = perm2[i][assignment[i,2]]
            perm2_saliency[i] = saliency[i][assignment[i,2]]
        
        #####
        # Saliency-guided sequential sampling
        #####
        #Eq.(4) in the main paper
        xyz_saliency = xyz_saliency/xyz_saliency.sum(-1, keepdim=True)
        anc_idx = torch.multinomial(xyz_saliency, 1, replacement=True)
        anchor_ori = xyz_new[torch.arange(B), anc_idx[:,0]]


        # print("anchor ori:",anchor_ori.shape)
        #cal distance and reweighting saliency map for Eq.(5) in the main paper
        sub = perm1_new - anchor_ori[:,None,:]
        dist = ((sub) ** 2).sum(2).sqrt()
        perm1_saliency = perm1_saliency * dist
        perm1_saliency = perm1_saliency/perm1_saliency.sum(-1, keepdim=True)
        # print("perm1_saliency:",perm1_saliency)
        
        #Eq.(5) in the main paper
        anc_idx2 = torch.multinomial(perm1_saliency, 1, replacement=True)
        anchor_perm1 = perm1_new[torch.arange(B),anc_idx2[:,0]]

        # print("anchor_perm:",anchor_perm1.shape)
        
        sub21 = (perm2_new - anchor_ori[:,None,:])
        dist21 = ((sub21) ** 2).sum(2).sqrt()
        sub22 = (perm2_new - anchor_perm1[:,None,:])
        dist22 = ((sub22) ** 2).sum(2).sqrt()
        perm2_saliency = perm2_saliency * dist21 + perm2_saliency * dist22
        perm2_saliency = perm2_saliency/perm2_saliency.sum(-1, keepdim=True)
        # print("perm2_saliency:",perm2_saliency)

        #Eq.(5) in the main paper
        anc_idx3 = torch.multinomial(perm2_saliency, 1, replacement=True)
        anchor_perm2 = perm2_new[torch.arange(B),anc_idx3[:,0]]

    
        # return
        #####
        # Shape-preserving continuous Mixup
        #####
        pi = self.dirich.sample((B,)).cuda()
        # print("pi sum",pi.sum(dim=1))


        # return
        sub_ori = xyz_new - anchor_ori[:,None,:]
        sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        #Eq.(6) for first sample
        ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
        
        sub_perm1 = perm1_new - anchor_perm1[:,None,:]
        sub_perm1 = ((sub_perm1) ** 2).sum(2).sqrt()
        #Eq.(6) for second sample
        ker_weight_perm1 = torch.exp(-0.5 * (sub_perm1 ** 2) / (self.sigma ** 2))  #(M,N)

        sub_perm2 = perm2_new - anchor_perm2[:,None,:]
        sub_perm2 = ((sub_perm2) ** 2).sum(2).sqrt()
        #Eq.(6) for third sample
        ker_weight_perm2 = torch.exp(-0.5 * (sub_perm2 ** 2) / (self.sigma ** 2))  #(M,N)

        
        # print("ker_weight_ori:",ker_weight_ori.shape)
        #Eq.(9)
        weight_ori = ker_weight_ori * pi[:,0][:,None]
        weight_perm1 = ker_weight_perm1 * pi[:,1][:,None]
        weight_perm2 = ker_weight_perm2 * pi[:,2][:,None]

        weight = (torch.cat([weight_ori[...,None],weight_perm1[...,None],weight_perm2[...,None]],-1)) + 1e-16
        weight = weight/weight.sum(-1)[...,None]
        # print("weight:",weight.shape)
        #Eq.(8) for new sample
        x = weight[:,:,0:1] * xyz_new + weight[:,:,1:2] * perm1_new + weight[:,:,2:] * perm2_new
        
        #Eq.(8) for new sample
        # x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        
        #Eq.(8) for new label
        target = weight.sum(1)
        target = target / target.sum(-1, keepdim=True)
        # print("label shape",label.shape)
        # print(self.num_class)
        label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_perm1_onehot = label_onehot[idxs1]
        label_perm2_onehot = label_onehot[idxs2]
        label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm1_onehot + target[:, 2, None] * label_perm2_onehot

        # print("label new shape:",label.shape)

        
        return x, label


io = IOStream('checkpoints/' + args.exp_name + '/run.log')
io.cprint(str(args))
if args.use_sgd:
    #print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
else:
    #print("Use Adam")
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


best_test_acc = 0
sagemix=SageMix(args,device,num_class=num_class)
scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
criterion = cal_loss_mix

# Check if multiple GPUs are available and wrap the model
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)



import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="ThreePointMix_nonhierarch",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "Pointnet++",
    "dataset": "MN40"
    }
)

best_test_acc = 0
for epoch in range(args.epochs):

    ####################
    # Train
    ####################
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for data, label in tqdm(train_loader):
        # start = time.time()
        data, label = data.to(device), label.to(device).squeeze()
        # print("data shape", data.shape)
        batch_size = data.size()[0]
        
        ####################
        # generate augmented sample
        ####################
        model.eval()
        data_var = Variable(data.permute(0,2,1), requires_grad=True)
        logits = model(data_var)
        loss = cal_loss(logits, label, smoothing=False)
        loss.backward()
        opt.zero_grad()
        saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
        data, label = sagemix.mix(data, label, saliency)

        # break
        
        mixed_saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
        # print("data shape", data.shape)
        model.train()
        # break
            
        opt.zero_grad()
        logits = model(data.permute(0,2,1))
        loss = criterion(logits, label)
        loss.backward()
        opt.step()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        # print('time of batch:', time.time() - start)
    # break 
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
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_three_point_nonhierarch.t7' % args.exp_name)
    wandb.log({
        "loss": loss, 
        "test_acc": test_acc,
        "test_avg_acc": avg_per_class_acc, 
        "best_test_acc": best_test_acc, 
    }, step=epoch)

    outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc,
                                                                            best_test_acc)
    io.cprint(outstr)