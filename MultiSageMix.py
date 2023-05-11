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
from emd_ import emd_module
import wandb
args = argparse.Namespace(batch_size=30, data='MN40', dropout=0.5, emb_dims=1024, epochs=200, eval=False, exp_name='MultiSageMix', k=20, lr=0.0001, model='pointnet', model_path='', momentum=0.9, no_cuda=False, num_points=1024, seed=1, sigma=-1, test_batch_size=16, theta=0.2, use_sgd=False)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    args.batch_size = 40
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#Try to load models
if args.model == 'pointnet':
    model = PointNet(args, num_class).to(device)
elif args.model == 'dgcnn':
    model = DGCNN(args, num_class).to(device)
else:
    raise Exception("Not implemented")


class SageMix:
    def __init__(self, args, num_class=40):
        self.num_class = num_class
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        self.beta = torch.distributions.beta.Beta(torch.tensor([args.theta]), torch.tensor([args.theta]))
        self.beta2 = torch.distributions.beta.Beta(torch.tensor([2*args.theta]), torch.tensor([args.theta]))

    
    def mix(self, xyz, label, saliency=None, mixing_idx=0, device='cuda:0'):
        """
        Args:
            xyz (B,N,3)
            label (B)
            saliency (B,N): Defaults to None.
        """        
        # #print("xyz shape", xyz.shape)
        B, N, _ = xyz.shape
        if mixing_idx == 0:
            idxs = torch.randperm(B)

            # #print(xyz)
            
            #Optimal assignment in Eq.(3)
            perm = xyz[idxs]
            
            _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
            ass = ass.long()
            # #print(ass)
            perm_new = torch.zeros_like(perm).to(device)#.cuda()
            perm_saliency = torch.zeros_like(saliency).to(device)#.cuda()
            
            # #print(ass,ass.shape)
            for i in range(B):
                perm_new[i] = perm[i][ass[i]]
                # #print(idxs)
                # #print(ass)
                # #print(saliency)
                # #print("idxs shape", idxs.shape)
                # #print("ass shape", ass.shape)
                # #print("saliency shape", saliency.shape)
                perm_saliency[i] = saliency[idxs][i][ass[i]]
            
            #####
            # Saliency-guided sequential sampling
            #####
            #Eq.(4) in the main paper
            saliency = saliency/saliency.sum(-1, keepdim=True)
            anc_idx = torch.multinomial(saliency, 1, replacement=True)
            anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]
            
            #cal distance and reweighting saliency map for Eq.(5) in the main paper
            sub = perm_new - anchor_ori[:,None,:]
            dist = ((sub) ** 2).sum(2).sqrt()
            perm_saliency = perm_saliency * dist
            perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
            
            #Eq.(5) in the main paper
            anc_idx2 = torch.multinomial(perm_saliency, 1, replacement=True)
            anchor_perm = perm_new[torch.arange(B),anc_idx2[:,0]]
                    
                    
            #####
            # Shape-preserving continuous Mixup
            #####
            alpha = self.beta.sample((B,)).to(device)#.cuda()
            sub_ori = xyz - anchor_ori[:,None,:]
            sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
            #Eq.(6) for first sample
            ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)
            
            sub_perm = perm_new - anchor_perm[:,None,:]
            sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
            #Eq.(6) for second sample
            ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)
            
            #Eq.(9)
            weight_ori = ker_weight_ori * alpha 
            weight_perm = ker_weight_perm * (1-alpha)
            weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
            weight = weight/weight.sum(-1)[...,None]

            #Eq.(8) for new sample
            x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
            
            #Eq.(8) for new label
            target = weight.sum(1)
            target = target / target.sum(-1, keepdim=True)
            
            label_onehot = torch.zeros(B, self.num_class).to(device).scatter(1, label.view(-1, 1), 1)
            label_perm_onehot = label_onehot[idxs]
            label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot
            return x, label
        
        else:
            # #print("xyz shape mixing 1", xyz.shape)
            B, N, _ = xyz.shape
            split_idx = int(B/2)
            # #print("split_idx", split_idx)
            # #print("saliency shape", saliency.shape)

            xyz1 = xyz[:split_idx]
            xyz2 = xyz[split_idx:]
            label1 = label[:split_idx]
            label2 = label[split_idx:]
            saliency1 = saliency[:split_idx]
            saliency2 = saliency[split_idx:]

            _, ass = self.EMD(xyz1, xyz2, 0.005, 500) # mapping
            ass = ass.long()

            #####
            # Saliency-guided sequential sampling
            #####
            #Eq.(4) in the main paper
            saliency1 = saliency1/saliency1.sum(-1, keepdim=True)
            anc_idx = torch.multinomial(saliency1, 1, replacement=True)
            anchor_ori = xyz1[torch.arange(split_idx), anc_idx[:,0]]

            #cal distance and reweighting saliency map for Eq.(5) in the main paper
            sub = xyz2 - anchor_ori[:,None,:]
            dist = ((sub) ** 2).sum(2).sqrt()
            # #print("saliency2 shape", saliency2.shape)
            # #print("dist shape", dist.shape)
            saliency2 = saliency2 * dist
            saliency2 = saliency2/saliency2.sum(-1, keepdim=True)
            
            #Eq.(5) in the main paper
            anc_idx2 = torch.multinomial(saliency2, 1, replacement=True)
            anchor_2 = xyz2[torch.arange(split_idx),anc_idx2[:,0]]

            alpha = self.beta.sample((split_idx,)).to(device)#.cuda()
            sub_ori = xyz1 - anchor_ori[:,None,:]
            sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
            #Eq.(6) for first sample
            ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)

            # #print("anchor_2 shape", anchor_2.shape)
            sub_perm = xyz2 - anchor_2[:,None,:]
            sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
            #Eq.(6) for second sample
            ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)

            #Eq.(9)
            weight_ori = ker_weight_ori * alpha
            weight_perm = ker_weight_perm * (1-alpha)
            weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16
            weight = weight/weight.sum(-1)[...,None]

            #Eq.(8) for new sample
            x = weight[:,:,0:1] * xyz1 + weight[:,:,1:] * xyz2

            #Eq.(8) for new label
            target = weight.sum(1)
            target = target / target.sum(-1, keepdim=True)

            label = target[:, 0, None] * label1 + target[:, 1, None] * label2



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
sagemix=SageMix(args, num_class)
scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
criterion = cal_loss_mix


def interleave(a, b):
    # Check that a and b have the same shape
    assert a.shape == b.shape, "Tensors must have the same shape"
    
    # Expand dimensions
    a = a.unsqueeze(1)
    b = b.unsqueeze(1)

    # Concatenate tensors
    c = torch.cat((a, b), dim=1)

    # Reshape tensor
    c = c.view(-1, *a.shape[2:])

    return c


wandb.init(
    # set the wandb project where this run will be logged
    project="ThreePointSageMixup",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "Pointnet++",
    "dataset": "MN40",
    "epochs": args.epochs,
    }
)

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
        data, label = data.to(device), label.to(device).squeeze()
        # #print("data shape", data)
        batch_size = data.size()[0]
        split_idx = int(batch_size * 1/2)
        data01 = data[:split_idx, :, :]
        label01 = label[:split_idx]
        data2 = data[split_idx:, :, :]
        label2 = label[split_idx:]
        
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
        # #print("saliency shape", saliency.shape)
        # #print("data01 shape", data01.shape)
        data_mix, label_mix = sagemix.mix(data01, label01, saliency[:split_idx,:], mixing_idx = 0)

        label2_onehot = torch.zeros(label2.shape[0], num_class).to(device).scatter(1, label2.view(-1, 1), 1)
        # #print("label2_onehot shape", label2_onehot.shape)
        # #print("label_mix shape", label_mix.shape)
        # label2_perm_onehot = label2_onehot[idxs]
        # label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot
        # #print("data_mix shape", data_mix.shape)
        # #print("data2 shape", data2.shape)
        # data_all = interleave(data_mix, data2)
        data_all = torch.cat((data_mix, data2), dim=0)
        # #print("data_all shape", data_all.shape)
        # label_all = interleave(label_mix, label2_onehot)
        label_all = torch.cat((label_mix, label2_onehot), dim=0)

        data_var = Variable(data_mix.permute(0,2,1), requires_grad=True)
        logits = model(data_var)
        loss_mix = criterion(logits, label_mix)
        loss_mix.backward()
        opt.zero_grad()
        saliency_mix = torch.sqrt(torch.mean(data_var.grad**2,1))

        

        # saliency_all = interleave(saliency_mix, saliency[split_idx:, :])
        saliency_all = torch.cat((saliency_mix, saliency[split_idx:,:]), dim=0)

        data_total_mix, label_total_mix = sagemix.mix(data_all, label_all, saliency_all, mixing_idx=1)
        
        model.train()
        # # break
            
        opt.zero_grad()
        # opt.zero_grad()
        logits = model(data_total_mix.permute(0,2,1))
        loss = criterion(logits, label_total_mix)
        loss.backward()
        opt.step()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        # logits3 = model(data3.permute(0,2,1))
        # loss3 = criterion(logits3, label3)
        # loss3.backward()
        # opt.step()
        # preds = logits3.max(dim=1)[1]
        # count += batch_size
        # train_loss += loss3.item() * batch_size
        
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
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_three_point.t7' % args.exp_name)

    wandb.log({
        "loss": loss, 
        "test_acc": test_acc,
        "test_avg_acc": avg_per_class_acc, 
        "best_test_acc": best_test_acc, 
        "epoch": epoch
    }, step=epoch)

    outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc,
                                                                            best_test_acc)
    io.cprint(outstr)
    




