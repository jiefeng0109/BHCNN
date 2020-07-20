# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:11:22 2018
Thanks for my girlfriend's support
@author: Jiantong Chen
"""
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data,windowFeature,one_hot,disorder,next_batch
from processing_library import first_layer,conv_layer_same,conv_layer_valid,contrary_one_hot
from processing_library import order_weight_fixed,index_band_selection,save_result


class HDCNN_model(object):
    
    def __init__(self,dim_input,num_classification,spatial_w,learn_rate,global_step,batch_size,all_step):
        
        self._dim_input=dim_input
        self._num_classification=num_classification
        self._spatial_w=spatial_w
        self._learn_rate=learn_rate
        self._global_step=global_step
        self._batch_size=batch_size
        self._ratio=tf.cast(self._global_step/all_step+0.8,dtype=tf.float32)
        self._ratio = tf.cond(self._ratio < 0.0, lambda:0.0, lambda:self._ratio) 
        self._ratio = tf.cond(self._ratio > 1.0, lambda:1.0, lambda:self._ratio) 
        
        self.config()
        loss=(1-self._ratio)*self.count_loss(mode='no select')+self._ratio*self.count_loss(mode='select')
        self.train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(loss, global_step)

        self.correct_prediction = tf.equal(tf.argmax(self._y,1), tf.argmax(self._y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
    def config(self):
        self._weights={'W1':tf.Variable(tf.truncated_normal([1,1,self._dim_input,1],stddev=0.1)),
         
                 'W2':tf.Variable(tf.truncated_normal([1,1,self._dim_input,32],stddev=0.1)),
                 
                 'W3':tf.Variable(tf.truncated_normal([3,3,self._dim_input,32],stddev=0.1)),
                                                      
                 'W4':tf.Variable(tf.truncated_normal([4,4,2*32,64],stddev=0.1)),
                 
                 'W5':tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.1)),
        
                 'W6':tf.Variable(tf.truncated_normal([3,3,128,256],stddev=0.1)),
        
                 'W7':tf.Variable(tf.truncated_normal([256,512],stddev=0.1)),
        
                 'W8':tf.Variable(tf.truncated_normal([512,self._num_classification],stddev=0.1)),
        
                 'W9':tf.Variable(tf.truncated_normal([2*32,self._num_classification],stddev=0.1)),
        
                 'W10':tf.Variable(tf.truncated_normal([128,self._num_classification],stddev=0.1)),                                    
                                                      
                 'W11':tf.Variable(tf.truncated_normal([256,self._num_classification],stddev=0.1)),
             
                 'W12':tf.Variable(tf.truncated_normal([512,self._num_classification],stddev=0.1))
                 }
        
        self._bias={'B1':tf.Variable(tf.constant(0.1,shape=[self._dim_input])),
              
              'B2':tf.Variable(tf.constant(0.1,shape=[32])),
              
              'B3':tf.Variable(tf.constant(0.1,shape=[32])),
              
              'B4':tf.Variable(tf.constant(0.1,shape=[64])),
                   
              'B5':tf.Variable(tf.constant(0.1,shape=[128])),
                  
              'B6':tf.Variable(tf.constant(0.1,shape=[256])),
        
              'B7':tf.Variable(tf.constant(0.1,shape=[512])),
                                           
              'B8':tf.Variable(tf.constant(0.1,shape=[self._num_classification])),                             
        
              'B9':tf.Variable(tf.constant(0.1,shape=[self._num_classification])),
                                    
              'B10':tf.Variable(tf.constant(0.1,shape=[self._num_classification])),  
        
              'B11':tf.Variable(tf.constant(0.1,shape=[self._num_classification])),
        
              'B12':tf.Variable(tf.constant(0.1,shape=[self._num_classification]))
              }
        
        self.x = tf.placeholder(tf.float32,[None,self._spatial_w,self._spatial_w,self._dim_input],name='x_input')
        self._y = tf.placeholder(tf.float32,[None,self._num_classification],name='y_output')
        self._x_reshape = tf.reshape(self.x,shape=[-1,self._spatial_w,self._spatial_w,self._dim_input])
        self._keep_prob = tf.placeholder(tf.float32)
        self._WWW=tf.placeholder(tf.float32,[1,1,self._dim_input,1],name='x_input')
    
    def count_loss(self,mode='select'):
            
        if mode == 'select':
            self._conv1 = first_layer(self._x_reshape,tf.multiply(self._weights['W1'],self._WWW),self._bias['B1'],[1,1,1,1])
        else:
            self._conv1 = first_layer(self._x_reshape, self._weights['W1'], self._bias['B1'], [1, 1, 1, 1])
        
        self._conv2 = conv_layer_same(self._conv1,self._weights['W2'],self._bias['B2'],[1,1,1,1])
        self._conv3 = conv_layer_same(self._conv1,self._weights['W3'],self._bias['B3'],[1,1,1,1])
        self._youknow=tf.concat([self._conv2,self._conv3],3)
        
        self._conv4 = conv_layer_valid(self._youknow,self._weights['W4'],self._bias['B4'],[1,1,1,1])
        
        self._pool5 = tf.nn.max_pool(self._conv4,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        self._conv6 = conv_layer_same(self._pool5,self._weights['W5'],self._bias['B5'],[1,1,1,1])
        
        self._dpt7 = tf.nn.dropout(self._conv6,self._keep_prob)
        
        self._pool8 = tf.nn.max_pool(self._dpt7,[1,2,2,1],[1,2,2,1], padding='SAME')
        
        self._conv9 = conv_layer_valid(self._pool8,self._weights['W6'],self._bias['B6'],[1,1,1,1])
        
        self._dpt10 = tf.nn.dropout(self._conv9,self._keep_prob)
        self._reshape = tf.reshape(self._dpt10,[-1,self._weights['W7'].get_shape().as_list()[0]])
        
        self._f11 = tf.nn.relu(tf.add(tf.matmul(self._reshape,self._weights['W7']),self._bias['B7']))
        
        self._f12 = tf.add(tf.matmul(self._f11,self._weights['W8']),self._bias['B8'])
        
        self._y_=tf.nn.softmax(self._f12)
             
        self._out1=tf.add(tf.matmul(tf.reshape(tf.slice(self._youknow, [0,int(self._spatial_w/2),int(self._spatial_w/2),0], [self._batch_size,1,1,2*32]),shape=[-1,2*32]),self._weights['W9']),self._bias['B9'])
        self._out2=tf.add(tf.matmul(tf.reshape(tf.slice(self._pool8, [0,int(3/2),int(3/2),0], [self._batch_size,1,1,128]),shape=[-1,128]),self._weights['W10']),self._bias['B10'])
        self._out3=tf.add(tf.matmul(self._reshape,self._weights['W11']),self._bias['B11'])
        self._out4 = tf.add(tf.matmul(self._f11,self._weights['W12']),self._bias['B12'])

        self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._f12, name=None))
        self._cross_entropy_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._out1, name=None))
        self._cross_entropy_yy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._out2, name=None))
        self._cross_entropy_yyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._out3, name=None))
        self._cross_entropy_yyyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._out4, name=None))
        
        self._loss=tf.add(tf.add(tf.add(tf.add(self._cross_entropy,0.5*self._cross_entropy_y),0.5*self._cross_entropy_yy),0.5*self._cross_entropy_yy),0.5*self._cross_entropy_yyyy)
        
        return self._loss

        
class HDCNN(object):
    
    def __init__(self,data_name,num_band_seclection,
                 batch_size=128,
                 display_step=100,
                 step=1,
                 spatial_w=15,
                 num_epoch=300):
        
        self._data_name=data_name
        self._num_band_seclection=num_band_seclection#要选择的波段数
        self._spatial_w =spatial_w#图像块大小
        self._batch_size = batch_size
        self._step = step
        self._index = self._batch_size
        self._display_step = display_step
        self._global_step=tf.Variable(self._step)
        self._learn_rate=tf.train.exponential_decay(0.05, self._global_step,100,0.8, staircase=False)#学习率
        self._num_epoch = num_epoch#训练循环次数       
        self.data_read(self._data_name)
        self.all_step=int(self._num_epoch*(self._X_train.shape[0]/self._batch_size))

        self._net=HDCNN_model(self._dim_input,self._num_classification,self._spatial_w,self._learn_rate,self._global_step,self._batch_size,self.all_step)
    
    def data_read(self,data_name):
        self._data_norm,self._labels_ori,self._x_train,self._y_train,self._train_loc,self._x_test,self._y_test,self._test_loc=load_data(data_name)
        self._dim_input = np.shape(self._data_norm)[2]
        self._num_classification = int(np.max(self._labels_ori))#类别数
        
        self._X_train = windowFeature(self._data_norm, self._train_loc, self._spatial_w)
        self._X_test = windowFeature(self._data_norm, self._test_loc, self._spatial_w)
        
        print('X_train shape:', self._X_train.shape)
        print('X_test shape:', self._X_test.shape)
        print(self._X_train.shape[0], 'train samples')
        print(self._X_test.shape[0], 'test samples')
        
        self._Y_train = one_hot(self._y_train,self._num_classification )
        self._Y_test = one_hot(self._y_test,self._num_classification )
        
        self._X_train,self._Y_train,self._train_loc=disorder(self._X_train,self._Y_train,self._train_loc)
        self._X_test,self._Y_test,self._test_loc=disorder(self._X_test,self._Y_test,self._test_loc)
    
    def get_oa(self,X_valid,Y_valid):
        size = np.shape(X_valid)
        num = size[0]
        index_all = 0
        step_ = 3000
        y_pred = []
        while index_all<num:
            if index_all + step_ > num:
                input = X_valid[index_all:, :, :, :]
            else:
                input = X_valid[index_all:(index_all+step_), :, :, :]
            index_all += step_
            temp1 = self._net._y_.eval(feed_dict={self._net.x: input,self._net._keep_prob:1.0,self._net._WWW:order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)})
            y_pred1=contrary_one_hot(temp1).astype('int32')
            y_pred.extend(y_pred1)
        y=contrary_one_hot(Y_valid).astype('int32')
        return y_pred,y
        
    def train_test(self):
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            self._epoch = 0
            self._time_train_start=time.clock()
            while self._epoch<self._num_epoch:
                self._batch_x,self._batch_y = next_batch(self._X_train,self._Y_train,self._index)
                sess.run(self._net.train_step,feed_dict={self._net.x: self._batch_x,
                         self._net._y: self._batch_y,
                         self._net._keep_prob:0.5,
                         self._net._WWW:order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)})
            
#                print(self._net._ratio.eval())
#                print(index_band_selection(order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)))
#                print(order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)[0,0,:,0])
#                print(self._net._weights['W1'].eval())
#                print(self._net._conv1.eval(feed_dict={self._net.x: self._batch_x,self._net._WWW:order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)})[0,:,:,0])
            
#                if self._step%self._display_step == 0:
#                    self._acc = sess.run(self._net.accuracy, feed_dict={self._net.x: self._batch_x, self._net._y: self._batch_y,
#                                                   self._net._keep_prob:1.0,self._net._WWW:order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)})
#                    print('step %d,training accuracy %f'%(self._step,self._acc))
#                    y_pr,y_tr = self.get_oa(self._X_test,self._Y_test)
#                    oa = accuracy_score(y_tr,y_pr)
#                    print('valid accuracy %f'%(oa))
                self._index = self._index+self._batch_size
                self._step += 1
                if self._index>self._X_train.shape[0]:
                    self._index = self._batch_size
                    self._epoch=self._epoch+1       
            self._time_train_end=time.clock()   
        
            print("Optimization Finished!")
        
            self._time_test_start=time.clock()
            self._y_pr,self._y_real = self.get_oa(self._X_test,self._Y_test)
            self._oa=accuracy_score(self._y_real,self._y_pr)
            self._per_class_acc=recall_score(self._y_real,self._y_pr,average=None)
            self._aa=np.mean(self._per_class_acc)
            self._kappa=cohen_kappa_score(self._y_real,self._y_pr)
            self._time_test_end=time.clock()
                
            print(self._per_class_acc)
            print(self._oa,self._aa,self._kappa)
            print((self._time_train_end-self._time_train_start),(self._time_test_end-self._time_test_start))
            

            self._band_loction=index_band_selection(order_weight_fixed(self._net._weights['W1'],self._num_band_seclection)) 
            self._num_band_seclection_now=len(self._band_loction)
            save_result('Indian_pines',self._oa,self._aa,self._kappa,self._num_band_seclection_now,self._band_loction,
                        self._per_class_acc,(self._time_train_end-self._time_train_start),(self._time_test_end-self._time_test_start))

    def plot(self,num):
        
        plot_max=np.zeros(np.shape(self._labels_ori))
        for i in range(self._X_train.shape[0]):
            plot_max[self._train_loc[0,i]][self._train_loc[1,i]]=self._labels_ori[self._train_loc[0,i]][self._train_loc[1,i]]
        for i in range(self._X_test.shape[0]):
            plot_max[self._test_loc[0,i]][self._test_loc[1,i]]=self._y_pr[i]
        
        path=os.getcwd()
        sio.savemat(path+'/plot/'+'PaviaU'+'_'+'plot'+str(num), {'plot_max':plot_max})
    