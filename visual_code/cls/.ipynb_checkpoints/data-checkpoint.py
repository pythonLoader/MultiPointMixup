#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', selfmix=False):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        
        
        #for self-mixup
        self.selfmix = selfmix
        
        if selfmix:
            order = self.label.argsort(0)[:,0]
            self.data = self.data[order]
            self.label = self.label[order]
            
            self.start_idxs = [0, 625, 731, 1246, 1419, 1991, 2326, 2390, 2587, 3476, 
                            3643, 3722, 3859, 4059, 4168, 4368, 4517, 4688, 4843, 4988, 
                            5112, 5261, 5545, 6010, 6210, 6298, 6529, 6768, 6872, 6987,
                            7115, 7795, 7919, 8009, 8401, 8564, 8908, 9175, 9650, 9737, 9840]
        
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        
        pointcloud2 = np.zeros_like(pointcloud)
        
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
            
            if self.selfmix:
                idx = np.random.randint(self.start_idxs[int(label)], self.start_idxs[int(label)+1])
                pointcloud2 = self.data[idx][:self.num_points]
                pointcloud2 = translate_pointcloud(pointcloud2)
                np.random.shuffle(pointcloud2)
                
            return pointcloud, label, pointcloud2
        else:
            return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='train', aug="default"):
        self.num_points = num_points
        self.partition = partition 
        self.aug = aug
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        
        if self.partition == 'train':
            file = os.path.join(DATA_DIR, "ScanObjectNN/main_split/training_objectdataset_augmentedrot_scale75.h5")
        else:
            file = os.path.join(DATA_DIR, "ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75.h5")

        f = h5py.File(file, 'r')
        self.data = f['data'][:].astype('float32')
        self.label = f['label'][:].astype('int64')
        f.close()

    def __getitem__(self, item):
        idx = np.arange(2048)
        np.random.shuffle(idx)
        idx = idx[:self.num_points]
        pointcloud = self.data[item][idx]
        label = self.label[item]
        
        if self.aug=="default":
            pointcloud -= pointcloud.mean(0)
            d = ((pointcloud**2).sum(-1)**(1./2)).max()
            pointcloud /= d
            
            if self.partition == 'train':
                pointcloud = rotate_pointcloud(pointcloud)
                pointcloud = jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05)
        elif self.aug=="MN40":
            if self.partition == 'train':
                pointcloud = translate_pointcloud(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

