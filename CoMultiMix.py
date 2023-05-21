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

import warnings
from match import get_onehot_matrix, mix_input
from math import ceil
import argparse

import importlib
import match
importlib.reload(match)
from match import get_onehot_matrix, mix_input

from emd_ import emd_module

# from SageMix import SageMix
from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN
from util import cal_loss, cal_loss_mix, IOStream
import gco
import torch.multiprocessing as mp

import os


import numpy as np
import torch
import torch.nn.functional as F
import warnings
from match import get_onehot_matrix, mix_input
from math import ceil

import importlib
import match
importlib.reload(match)
from match import get_onehot_matrix, mix_input

importlib.reload(gco)
import gco
# from match import get_onehot_matrix, mix_input

warnings.filterwarnings("ignore")

def align_batch(xyz,device,num_class=40,saliency=None):
    EMD = emd_module.emdModule()
    B, N, _ = xyz.shape
    
    idxs = torch.randperm(B)

    #Optimal assignment in Eq.(3)
    # perm = xyz[idxs]
    dist_mat = torch.empty(B, B, 1024)
    ass_mat = torch.empty(B,B,1024)
    dist_mat = dist_mat.to(device)
    
    # print("Starting to compute optimal assignment (Heuristic-1)")
    for idx,point in enumerate(xyz):
        # perm = torch.tensor([point for x in range(B))
        # print(point.shape)
        perm = point.repeat(B,1)
        # print(perm.shape)

        perm  = perm.reshape(perm.shape[0]//1024,1024,3)
        
        dist, ass = EMD(xyz, perm, 0.005, 500) # mapping
                # 32,1024
        dist_mat[idx] = dist
        ass_mat[idx] = ass
    
    # print(dist_mat.shape)
    dist_mat = torch.norm(dist_mat,dim=2)
    avg_alignment_dist = torch.mean(dist_mat,dim=0)
    
    idx = torch.argmin(avg_alignment_dist).item()
    
    ass = ass_mat[idx]
    
    ass = ass.long()
    perm_new = torch.zeros_like(perm).to(device)
    
    perm = xyz.clone()
    
    for i in range(B):
        perm_new[i] = perm[i][ass[i]]
    return ass,perm_new,dist_mat


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

def mixup_process_worker(out: torch.Tensor,
                         target_reweighted: torch.Tensor,
                         hidden=0,
                         args=None,
                         sc: torch.Tensor = None,
                         A_dist: torch.Tensor = None,
                         debug=False):
    m_block_num = args.m_block_num
    m_part = args.m_part


    if A_dist is None:
        A_dist = torch.eye(batch_size, device=out.device)

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = 8
    out_list = []
    target_list = []


    
    with torch.no_grad():
        sc_part = sc
        A_dist_part = A_dist

        n_input = sc.shape[0]
        sc_norm = sc/torch.sum(sc, dim=1).view(-1,1)
        
        cost_matrix = -sc_norm
        # print("cost_matrix shape:",cost_matrix.shape)
        # print(cost_matrix.shape)

        A_base = torch.eye(n_input, device=out.device)
        A_dist_part = A_dist_part / torch.sum(A_dist_part) * n_input
        A = (1 - args.m_omega) * A_base + args.m_omega * A_dist_part
        mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                        A,
                                        n_output=3,
                                        beta=args.m_beta,
                                        gamma=args.m_gamma,
                                        eta=args.m_eta,
                                        mixup_alpha=args.mixup_alpha,
                                        thres=args.m_thres,
                                        thres_type=args.m_thres_type,
                                        set_resolve=args.set_resolve,
                                        niter=args.m_niter,
                                        device='cuda')
    output_part, target_part = mix_input(mask_onehot, out,
                                             target_reweighted)

    out_list = output_part
    target_list = target_part


    return output_part, target_part

def mixup_process_worker_wrapper(q_input: mp.Queue, q_output: mp.Queue, device: int):
    """
    :param q_input:		input queue
    :param q_output:	output queue
    :param device:		running gpu device
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    print(f"Process generated with cuda:{device}")
    device = torch.device(f"cuda:{device}")
    while True:
        # Get and load on gpu
        out, target_reweighted, hidden, args, sc, A_dist, debug = q_input.get()
        out = out.to(device)
        target_reweighted = target_reweighted.to(device)
        sc = sc.to(device)
        A_dist = A_dist.to(device)

        # Run
        out, target_reweighted = mixup_process_worker(out, target_reweighted, hidden, args, sc,
                                                      A_dist, debug)
        # To cpu and return
        out = out.cpu()
        target_reweighted = target_reweighted.cpu()
        q_output.put([out, target_reweighted])

class MixupProcessWorker:
    def __init__(self, device: int):
        """
        :param device: gpu device id
        """
        self.q_input = mp.Queue()
        self.q_output = mp.Queue()
        self.worker = mp.Process(target=mixup_process_worker_wrapper,
                                 args=[self.q_input, self.q_output, device])
        self.worker.deamon = True
        self.worker.start()

    def start(self,
              out: torch.Tensor,
              target_reweighted: torch.Tensor,
              hidden=0,
              args=None,
              sc: torch.Tensor = None,
              A_dist: torch.Tensor = None,
              debug=True):
        self.q_input.put([out, target_reweighted, hidden, args, sc, A_dist, debug])

    def join(self):
        input, target = self.q_output.get()
        return input, target

    def close(self):
        self.worker.terminate()


class MixupProcessParallel:
    def __init__(self, part, batch_size, num_gpu=1):
        """
        :param part:
        :param batch_size:
        :param num_gpu:
        """
        self.part = part
        self.batch_size = batch_size
        self.n_workers = ceil(batch_size / part)
        self.workers = [MixupProcessWorker(device=i % num_gpu) for i in range(self.n_workers)]

    def __call__(self,
                 out: torch.Tensor,
                 target_reweighted: torch.Tensor,
                 hidden=0,
                 args=None,
                 sc: torch.Tensor = None,
                 A_dist: torch.Tensor = None,
                 debug=False):
        '''
        :param out:					cpu tensor
        :param target_reweighted: 	cpu tensor
        :param hidden:
        :param args:				cpu args
        :param sc: 					cpu tensor
        :param A_dist: 				cpu tensor
        :param debug:
        :return:					out, target_reweighted (cpu tensor)
        '''

        for idx in range(self.n_workers):
            self.workers[idx].start(
                out[idx * self.part:(idx + 1) * self.part].contiguous(),
                target_reweighted[idx * self.part:(idx + 1) * self.part].contiguous(), hidden, args,
                sc[idx * self.part:(idx + 1) * self.part].contiguous(),
                A_dist[idx * self.part:(idx + 1) * self.part,
                       idx * self.part:(idx + 1) * self.part].contiguous(), debug)
        # join
        out_list = []
        target_list = []
        for idx in range(self.n_workers):
            out, target = self.workers[idx].join()
            out_list.append(out)
            target_list.append(target)

        return torch.cat(out_list), torch.cat(target_list)

    def close(self):
        for w in self.workers:
            w.close()

# try:
#     mp.set_start_method('spawn', force=True)
#     print("spawned")
# except RuntimeError:
#     pass

if __name__ ==  '__main__':
    try:
        mp.set_start_method('spawn',force=True)
        print("spawned")
    except RuntimeError:
        pass
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = argparse.Namespace(batch_size=32, data='MN40', dropout=0.5, emb_dims=1024, 
                            epochs=50, eval=False, exp_name='CoMultiMixParallel', k=20, lr=0.0001, model='pointnet', model_path='', 
                            momentum=0.9, no_cuda=False, num_points=1024, seed=1, sigma=-1, test_batch_size=16, theta=0.2, use_sgd=True)
                            
    num_points = 1024
    dataset = ModelNet40(partition='train', num_points=num_points)
    batch_size=args.batch_size

    test_batch_size = args.test_batch_size
    train_loader = DataLoader(dataset, num_workers=8,
                            batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points), num_workers=8,
                            batch_size=test_batch_size, shuffle=True, drop_last=False)
    num_class=40

    args2 = {'arch': 'preactresnet18', 'batch_size': 32, 
            'clean_lam': 1.0, 'comix': True, 
            'data_dir': './data/cifar100/', 'dataset': 'cifar100', 
            'decay': 0.0001, 'dropout': False, 'epochs': 300, 
            'evaluate': True, 'gammas': [0.1, 0.1], 'initial_channels': 64, 
            'labels_per_class': 500, 'learning_rate': 0.2, 
            'log_off': True, 'm_beta': 0.32, 
            'm_block_num': 4, 'm_eta': 0.05, 
            'm_gamma': 1.0, 'm_niter': 4, 'm_omega': 0.001, 
            'm_part': 20, 'm_thres': 0.83, 
            'm_thres_type': 'hard', 
            'mixup_alpha': 2.0, 
            'momentum': 0.9, 'ngpu': 1, 
            'parallel': False, 'print_freq': 100, 
            'resume': './checkpoint/cifar100_preactresnet18_eph300_comixup/checkpoint.pth.tar', 
            'root_dir': 'experiments', 'schedule': [100, 200], 'seed': 0, 
            'set_resolve': True, 'start_epoch': 0, 'tag': '', 
            'use_cuda': True, 'valid_labels_per_class': 0, 'workers': 0}

    args2 = argparse.Namespace(**args2)


    device = torch.device("cuda")

    model = PointNet(args, num_class).to(device)
    model = nn.DataParallel(model,device_ids=[0,1,2,3])

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
        

    # sagemix = SageMix(args, device, num_class)
    criterion = cal_loss_mix


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
            data, label = data.to(device), label.to(device).squeeze()
            # print(data)
            batch_size = data.size()[0]
            
            ####################
            # generate augmented sample
            ####################
            model.eval()
            # print(data.permute(0,2,1).shape)
            data_var = Variable(data.permute(0,2,1), requires_grad=True)
            logits = model(data_var)
            loss = cal_loss(logits, label, smoothing=False)
            loss.backward()
            opt.zero_grad()
            saliency = torch.sqrt(torch.mean(data_var.grad**2,1))
        
            _,perm_new,_ = align_batch(data,device = device,saliency=saliency) 
            target_reweighted = F.one_hot(label, num_classes=num_class).float()
            

            with torch.no_grad():
                # print(saliency.shape)
                sc = saliency.unsqueeze(1)
                # print("sc:",sc.shape)
                z = F.avg_pool1d(sc, kernel_size=8, stride=1)
                # print("z:",z.shape)
                z_reshape = z.reshape(args.batch_size, -1)
                # print("z_reshape:",z_reshape.shape)
                z_idx_1d = torch.argmax(z_reshape, dim=1)
                z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
                z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
                z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
                A_dist = distance(z_idx_2d, dist_type='l1')

            out0 = perm_new
            target_reweighted0 = target_reweighted
            
            

            
            # Parallel mixup wrapper
            mpp = MixupProcessParallel(args2.m_part, args.batch_size, num_gpu=1)
            # print("running multiprocess worker")
            # # For cuda initialize
            # torch.ones(3).cuda()
            # for iter in tqdm(range(1), desc="initialize"):
            #     out, target_reweighted = mpp(out0,
            #                                 target_reweighted0,
            #                                 args=args2,
            #                                 sc=saliency,
            #                                 A_dist=A_dist,
            #                                 debug=True)

            # Parallel run
            # for iter in tqdm(range(100), desc="parallel"):
            out, target_reweighted = mpp(out0,
                                        target_reweighted0,
                                        args=args2,
                                        sc=saliency,
                                        A_dist=A_dist,
                                        debug=True)

            # print((d["out"].cpu() == out.cpu()).float().mean())
            # print((d["target_reweighted"].cpu() == target_reweighted.cpu()).float().mean())

            # # Original run
            # out0cuda = out0.cuda()
            # target_reweighted0cuda = target_reweighted0.cuda()
            # sccuda = sc.cuda()
            # A_distcuda = A_dist.cuda()
            # for iter in tqdm(range(100), desc="original"):
            #     out, target_reweighted = mixup_process(out0cuda,
            #                                         target_reweighted0cuda,
            #                                         args=args,
            #                                         sc=sccuda,
            #                                         A_dist=A_distcuda,
            #                                         debug=True)
            
            # out, target_reweighted = mixup_process(perm_new,
            #                                         target_reweighted,
            #                                         args=args2,
            #                                         sc=saliency,
            #                                         A_dist=A_dist)
            print("finished multiprocess worker")
            # print(out.dtype)
            out = out.to(device)
            target_reweighted = target_reweighted.to(device)

            model.train()
                
            opt.zero_grad()
            logits = model(out.permute(0,2,1))
            loss = criterion(logits, target_reweighted)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            
        scheduler.step()
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        print(outstr)
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
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                                test_loss*1.0/count,
                                                                                test_acc,
                                                                                avg_per_class_acc,
                                                                                best_test_acc)

        # io.cprint(outstr)