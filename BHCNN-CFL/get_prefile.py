# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import numpy as np
import scipy.io as sio
import os
from sklearn import preprocessing

class Get_prefile(object):
    
    def __init__(self,data_name,ratio):
        self._data_name=data_name
        self._ratio=ratio
        self._file_name=self._data_name+'_pre'
        self._path_in = os.getcwd()+'/data/'+self._data_name
        self._path_out = os.getcwd()+'/data/'+self._data_name+'/'+self._file_name
    
    def get_prefile(self):
    
        self.readData()
        self.normalizeData()
        self.selectTrainTest()
        self.save()
        
    def readData(self):
        if self._data_name == 'Indian_pines':
            self._data = sio.loadmat(self._path_in+'/Indian_pines_corrected.mat')['indian_pines_corrected']
            self._labels = sio.loadmat(self._path_in+'/Indian_pines_gt.mat')['indian_pines_gt']
        elif self._data_name == 'PaviaU':
            self._data = sio.loadmat(self._path_in+'/PaviaU.mat')['paviaU']
            self._labels = sio.loadmat(self._path_in+'/PaviaU_gt.mat')['paviaU_gt']
        elif self._data_name == 'KSC':
            self._data = sio.loadmat(self._path_in+'/KSC.mat')['KSC']
            self._labels = sio.loadmat(self._path_in+'/KSC_gt.mat')['KSC_gt']
        elif self._data_name == 'Salinas':
            self._data = sio.loadmat(self._path_in+'/Salinas_corrected.mat')['salinas_corrected']
            self._labels = sio.loadmat(self._path_in+'/Salinas_gt.mat')['salinas_gt']
        elif self._data_name == 'washington':
            self._data = sio.loadmat(self._path_in+'/washington.mat')['washington_datax']
            self._labels = sio.loadmat(self._path_in+'/washington_gt.mat')['washington_labelx']
        elif self._data_name == 'Houston':
            self._data = sio.loadmat(self._path_in+'/Houstondata.mat')['Houstondata']
            self._labels = sio.loadmat(self._path_in+'/Houstonlabel.mat')['Houstonlabel']
        self._data = np.float64(self._data )
        self._labels = np.array(self._labels).astype(float)

    def normalizeData(self):
        ''' 原始数据归一化处理（每条） '''
        self._data_norm = np.zeros(np.shape(self._data))
        for i in range(np.shape(self._data)[0]):
            for j in range(np.shape(self._data)[1]):
                self._data_norm[i,j,:] = preprocessing.normalize(self._data[i,j,:].reshape(1,-1))[0]
    
    def selectTrainTest(self):
        ''' 从所有类中每类选取训练样本和测试样本 '''
        self._c = int(self._labels.max())
        self._x = np.array([], dtype=float).reshape(-1, self._data.shape[2])  # 训练样本
        self._xb = []
        self._x_loc1 = []
        self._x_loc2 = []
        self._x_loc = []
        self._y = np.array([], dtype=float).reshape(-1, self._data.shape[2])
        self._yb = []
        self._y_loc1 = []
        self._y_loc2 = []
        self._y_loc = []
        for i in range(1, self._c+1):
        #i = 1
            self._loc1, self._loc2 = np.where(self._labels == i)
            self._num = len(self._loc1)
            self._order = np.random.permutation(range(self._num))
            self._loc1 = self._loc1[self._order]
            self._loc2 = self._loc2[self._order]
            self._num1 = int(np.round(self._num*self._ratio))
            self._x = np.vstack([self._x, self._data[self._loc1[:self._num1], self._loc2[:self._num1], :]])
            self._y = np.vstack([self._y, self._data[self._loc1[self._num1:], self._loc2[self._num1:], :]])
            self._xb.extend([i]*self._num1)
            self._yb.extend([i]*(self._num-self._num1))
            self._x_loc1.extend(self._loc1[:self._num1])
            self._x_loc2.extend(self._loc2[:self._num1])
            self._y_loc1.extend(self._loc1[self._num1:])
            self._y_loc2.extend(self._loc2[self._num1:])
            self._x_loc = np.vstack([self._x_loc1, self._x_loc2])
            self._y_loc = np.vstack([self._y_loc1, self._y_loc2])

    def save(self):
        sio.savemat(self._path_out, {'train_x':self._x,'train_y':self._xb, 'train_loc':self._x_loc, 'test_x':self._y,
                    'test_y':self._yb, 'test_loc':self._y_loc, 'data_norm':self._data_norm,'labels_ori':self._labels})

