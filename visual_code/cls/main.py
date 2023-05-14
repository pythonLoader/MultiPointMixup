#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

import wandb
from datetime import datetime
from tqdm import tqdm

import _init_path
from util import cal_loss, cal_loss_mix, IOStream
from cls.data import ModelNet40, ScanObjectNN
from cls.model_mixup import PointNet, DGCNN, Pointnet2_MSG
from torch.autograd import Variable

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main.py outputs'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    if args.data == 'MN40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=4,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 40
    elif args.data == 'SONN_EASY':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="easy"), num_workers=4,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="easy"), num_workers=4,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    elif args.data == 'SONN_HARD':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="hard"), num_workers=4,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="hard"), num_workers=4,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15

    device = torch.device("cuda" if args.cuda else "cpu")

        
    if args.model == 'pointnet':
        model = PointNet(args, num_class).to(device)
    elif args.model == 'pointnet2_MSG':
        model = Pointnet2_MSG(args, num_class).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.optim=="sgd":
        print("Use SGD")
        args.lr = args.lr*100
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    elif args.optim=="adam":
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr*0.01)
 
    
    if args.kermix:
        criterion = cal_loss_mix
        criterion2 = cal_loss
    else:
        criterion = cal_loss
        
    if args.wandb:
        wandb.init(project="NIPS2022_rebuttal", name=args.exp_name + datetime.now().strftime('-%Y/%m/%d_%H:%M:%S'))
        wandb.config.update(args)

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
        for data, label in (pbar := tqdm(train_loader)):
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            if args.kermix:
                model.eval()
                data_var = Variable(data.permute(0, 2, 1), requires_grad=True)
                logits, _ = model(data_var)
                loss = criterion2(logits, label, smoothing=False)
                loss.backward()
                opt.zero_grad()
                model.train()
                saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
                
                #### PointCloud Saliency Map
                # print("Test")

                # with torch.no_grad():
                #     sphere_core = data.median(axis=1, keepdims=True)[0]
                #     sphere_r = torch.sqrt(torch.sum(torch.square(data - sphere_core), axis=2)) ## BxN
                    
                #     sphere_axis = data - sphere_core ## BxNx3

                #     saliency = (torch.multiply(torch.sum(torch.multiply(data_var.grad.permute(0,2,1), sphere_axis), axis=2), torch.pow(sphere_r, 6)))
                #     saliency -= torch.min(saliency,1, keepdims=True)[0] 

                ####
                
            else:
                saliency = None
                opt.zero_grad()
    
         
            logits, label = model(data.permute(0,2,1), label, saliency=saliency)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()

        
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = train_loss*1.0/count
        # train_acc =  metrics.accuracy_score(train_true, train_pred)
        # train_avg_acc = metrics.balanced_accuracy_score (train_true, train_pred)
        # outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
        #                                                                          train_loss,
        #                                                                          train_acc,
        #                                                                          train_avg_acc)
        outstr = 'Train %d, loss: %.6f' % (epoch,train_loss)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        # test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                logits, _ = model(data)
                
            preds = logits.max(dim=1)[1]
            count += batch_size
            
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        # test_loss = test_loss*1.0/count
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                  test_acc,
                                                                  test_avg_acc)
        io.cprint(outstr)
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_avg_acc = test_avg_acc
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)
            
        log_dict = {"train_loss" : train_loss,
                "test_acc" : test_acc,
                "test_avg_acc" : test_avg_acc,
                "best_test_acc" : best_test_acc,
                "best_test_avg_acc " : best_test_avg_acc}

        if args.wandb:
            wandb.log(log_dict)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
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
                        choices=['pointnet', 'dgcnn', 'pointnet2_MSG'],
                        help='Model to use, [pointnet, dgcnn, pointnet2_MSG]')
    parser.add_argument('--data', type=str, default='MN40', metavar='N',
                        choices=['MN40', 'SONN_EASY', 'SONN_HARD'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--optim', type=str, default="sgd",
                        choices=['sgd', 'adam'],
                        help='Optimizer, [sgd, adam]')
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
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--kermix", action='store_true')   
    parser.add_argument("--basemix", action='store_true') 
    parser.add_argument('--sigma', type=float, default=0.5) 
    parser.add_argument('--beta', type=float, default=0.2) 
    
    args = parser.parse_args()
    _init_()
    
    io = IOStream('outputs/' + args.exp_name + '/run.log')
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
