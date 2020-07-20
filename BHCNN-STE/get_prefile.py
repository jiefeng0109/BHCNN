# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing

def readData(data_name):
    ''' 读取原始数据和标准类标 '''
    path = os.getcwd()+'/data/'+data_name
    if data_name == 'Indian_pines':
        data = sio.loadmat(path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == 'PaviaU':
        data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
        labels = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == 'KSC':
        data = sio.loadmat(path+'/KSC.mat')['KSC']
        labels = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']
    elif data_name == 'Salinas':
        data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
    elif data_name == 'washington':
        data = sio.loadmat(path+'/washington.mat')['washington_datax']
        labels = sio.loadmat(path+'/washington_gt.mat')['washington_labelx']
    elif data_name == 'Houston':
        data = sio.loadmat(path+'/Houstondata.mat')['Houstondata']
        labels = sio.loadmat(path+'/Houstonlabel.mat')['Houstonlabel']
    data = np.float64(data)
    labels = np.array(labels).astype(float)
    return data, labels

def normalizeData(data):
    ''' 原始数据归一化处理（每条） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data_norm[i,j,:] = preprocessing.normalize(data[i,j,:].reshape(1,-1))[0]
    return data_norm
    
def selectTrainTest(data, labels, p):
    ''' 从所有类中每类选取训练样本和测试样本 '''
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 训练样本
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []
    for i in range(1, c+1):
    #i = 1
        loc1, loc2 = np.where(labels == i)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]
        num1 = int(np.round(num*p))
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])
        y = np.vstack([y, data[loc1[num1:], loc2[num1:], :]])
        xb.extend([i]*num1)
        yb.extend([i]*(num-num1))
        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])
        y_loc1.extend(loc1[num1:])
        y_loc2.extend(loc2[num1:])
        x_loc = np.vstack([x_loc1, x_loc2])
        y_loc = np.vstack([y_loc1, y_loc2])
    return x, xb, x_loc, y, yb, y_loc
  
if __name__ == '__main__':

#    data_name = 'Indian_pines'
#    data_name = 'KSC'
#    data_name = 'Salinas'
    data_name = 'PaviaU' 
 #   data_name = 'Houston'

    data_ori, labels_ori = readData(data_name)
    data_norm = normalizeData(data_ori)
    if data_name == 'Indian_pines':
        p = 0.05
    elif data_name == 'PaviaU':
        p = 0.03
    elif data_name == 'KSC':
        p = 0.05
    elif data_name == 'Salinas':
        p = 0.01
    elif data_name == 'washington':
        p = 0.05
    elif data_name == 'Houston':
        p = 0.05
    train_x, train_y, train_loc, test_x, test_y, test_loc = selectTrainTest(data_norm, labels_ori, p)
    
    path = os.getcwd()
    sio.savemat(path+'/data/'+data_name+'/'+data_name+'_pre', {'train_x':train_x,
                'train_y':train_y, 'train_loc':train_loc, 'test_x':test_x,
                'test_y':test_y, 'test_loc':test_loc, 'data_norm':data_norm,
                'labels_ori':labels_ori})
