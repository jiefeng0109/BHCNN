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
###############################################################################
data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc=load_data('Indian_pines')

dim_input = np.shape(data_norm)[2]
batch_size = 128
display_step = 100
step = 1
index = batch_size

num_classification = int(np.max(labels_ori))#类别数
w =15#图像块大小
global_step=tf.Variable(step)
learn_rate=tf.train.exponential_decay(0.05, global_step,100,0.8, staircase=False)#学习率
num_epoch = 300#训练循环次数
num_band_seclection=30#要选择的波段数
#REGULARIZATION_RATE = 0.0001 # 正则化项的权重系数
###############################################################################
X_train = windowFeature(data_norm, train_loc, w)
X_test = windowFeature(data_norm, test_loc, w)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = one_hot(y_train,num_classification )
Y_test = one_hot(y_test,num_classification )

X_train,Y_train,train_loc=disorder(X_train,Y_train,train_loc)
X_test,Y_test,test_loc=disorder(X_test,Y_test,test_loc)
###############################################################################
weights={'W1':tf.Variable(tf.truncated_normal([1,1,dim_input,1],stddev=0.1)),
         
         'W2':tf.Variable(tf.truncated_normal([1,1,dim_input,32],stddev=0.1)),
         
         'W3':tf.Variable(tf.truncated_normal([3,3,dim_input,32],stddev=0.1)),
                                              
         'W4':tf.Variable(tf.truncated_normal([4,4,2*32,64],stddev=0.1)),
         
         'W5':tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.1)),

         'W6':tf.Variable(tf.truncated_normal([3,3,128,256],stddev=0.1)),

         'W7':tf.Variable(tf.truncated_normal([256,512],stddev=0.1)),

         'W8':tf.Variable(tf.truncated_normal([512,num_classification],stddev=0.1)),

         'W9':tf.Variable(tf.truncated_normal([2*32,num_classification],stddev=0.1)),

         'W10':tf.Variable(tf.truncated_normal([128,num_classification],stddev=0.1)),                                    
                                              
         'W11':tf.Variable(tf.truncated_normal([256,num_classification],stddev=0.1)),
     
         'W12':tf.Variable(tf.truncated_normal([512,num_classification],stddev=0.1))
         }

bias={'B1':tf.Variable(tf.constant(0.1,shape=[dim_input])),
      
      'B2':tf.Variable(tf.constant(0.1,shape=[32])),
      
      'B3':tf.Variable(tf.constant(0.1,shape=[32])),
      
      'B4':tf.Variable(tf.constant(0.1,shape=[64])),
           
      'B5':tf.Variable(tf.constant(0.1,shape=[128])),
          
      'B6':tf.Variable(tf.constant(0.1,shape=[256])),

      'B7':tf.Variable(tf.constant(0.1,shape=[512])),
                                   
      'B8':tf.Variable(tf.constant(0.1,shape=[num_classification])),                             

      'B9':tf.Variable(tf.constant(0.1,shape=[num_classification])),
                            
      'B10':tf.Variable(tf.constant(0.1,shape=[num_classification])),  

      'B11':tf.Variable(tf.constant(0.1,shape=[num_classification])),

      'B12':tf.Variable(tf.constant(0.1,shape=[num_classification]))
      }
###############################################################################
x = tf.placeholder(tf.float32,[None,w,w,dim_input],name='x_input')
y = tf.placeholder(tf.float32,[None,num_classification],name='y_output')
x_reshape = tf.reshape(x,shape=[-1,w,w,dim_input])
keep_prob = tf.placeholder(tf.float32)

WWW=tf.placeholder(tf.float32,[1,1,dim_input,1],name='x_input')
conv1 = first_layer(x_reshape,WWW,bias['B1'],[1,1,1,1])

conv2 = conv_layer_same(conv1,weights['W2'],bias['B2'],[1,1,1,1])
conv3 = conv_layer_same(conv1,weights['W3'],bias['B3'],[1,1,1,1])
youknow=tf.concat([conv2,conv3],3)

conv4 = conv_layer_valid(youknow,weights['W4'],bias['B4'],[1,1,1,1])

pool5 = tf.nn.max_pool(conv4,[1,2,2,1],[1,2,2,1],padding='SAME')

conv6 = conv_layer_same(pool5,weights['W5'],bias['B5'],[1,1,1,1])

dpt7 = tf.nn.dropout(conv6,keep_prob)

pool8 = tf.nn.max_pool(dpt7,[1,2,2,1],[1,2,2,1], padding='SAME')

conv9 = conv_layer_valid(pool8,weights['W6'],bias['B6'],[1,1,1,1])

dpt10 = tf.nn.dropout(conv9,keep_prob)
reshape = tf.reshape(dpt10,[-1,weights['W7'].get_shape().as_list()[0]])

f11 = tf.nn.relu(tf.add(tf.matmul(reshape,weights['W7']),bias['B7']))

f12 = tf.add(tf.matmul(f11,weights['W8']),bias['B8'])

y_=tf.nn.softmax(f12)

out1=tf.add(tf.matmul(tf.reshape(tf.slice(youknow, [0,int(w/2),int(w/2),0], [batch_size,1,1,2*32]),shape=[-1,2*32]),weights['W9']),bias['B9'])
out2=tf.add(tf.matmul(tf.reshape(tf.slice(pool8, [0,int(3/2),int(3/2),0], [batch_size,1,1,128]),shape=[-1,128]),weights['W10']),bias['B10'])
out3=tf.add(tf.matmul(reshape,weights['W11']),bias['B11'])
out4 = tf.add(tf.matmul(f11,weights['W12']),bias['B12'])

#writer = tf.summary.FileWriter('D:/ten',tf.get_default_graph())
#writer.close()
###############################################################################
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f12, name=None))
cross_entropy_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out1, name=None))
cross_entropy_yy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out2, name=None))
cross_entropy_yyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out3, name=None))
cross_entropy_yyyy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out4, name=None))

#regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#regularization = regularizer(regularizer(weights['W2'])+regularizer(weights['W3'])+regularizer(weights['W4'])+regularizer(weights['W5'])+regularizer(weights['W6']))
loss=tf.add(tf.add(tf.add(tf.add(cross_entropy,0.5*cross_entropy_y),0.5*cross_entropy_yy),0.5*cross_entropy_yy),0.5*cross_entropy_yyyy)
train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(loss,global_step) 

op=tf.assign(weights['W1'],tf.subtract(weights['W1'],tf.multiply(learn_rate,tf.gradients(loss,WWW)[0])))
###############################################################################
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.global_variables_initializer()
###############################################################################    
def get_oa(X_valid,Y_valid):
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
        temp1 = y_.eval(feed_dict={x: input,keep_prob:1.0,WWW:order_weight_fixed(weights['W1'],num_band_seclection)})
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
    y=contrary_one_hot(Y_valid).astype('int32')
    return y_pred,y
###############################################################################
with tf.Session() as sess:
    
    sess.run(init)
    epoch = 0
    time_train_start=time.clock()
    while epoch<num_epoch:
        batch_x,batch_y = next_batch(X_train,Y_train,index)
        sess.run([op,train_step],feed_dict={x: batch_x, y: batch_y,keep_prob:0.5,WWW:order_weight_fixed(weights['W1'],num_band_seclection)})
    
#        print(index_band_selection(order_weight_fixed(weights['W1'],num_band_seclection)))
#        print(order_weight_fixed(weights['W1'],num_band_seclection)[0,0,:,0])
#        print(conv1.eval(feed_dict={x: batch_x,W1:order_weight_fixed(weights['W1'],num_band_seclection)})[0,:,:,0])
    
#        if step%display_step == 0:
#            cros,cros_yy,los,acc = sess.run([cross_entropy,cross_entropy_yy,loss,accuracy], feed_dict={x: batch_x, y: batch_y,keep_prob:1.0,WWW:order_weight_fixed(weights['W1'],num_band_seclection)})
#            print('step %d,training accuracy %f'%(step,acc))
#            print('loss %f,cross_entropy %f,cross_entropy_yy %f'%(los,cros,cros_yy))
#            y_pr,y_tr = get_oa(X_test,Y_test)
#            oa = accuracy_score(y_tr,y_pr)
#            print('valid accuracy %f'%(oa))
        index = index+batch_size
        step += 1
        if index>X_train.shape[0]:
            index = batch_size
            epoch=epoch+1       
    time_train_end=time.clock()   

    print("Optimization Finished!")
    band_loction=index_band_selection(order_weight_fixed(weights['W1'],num_band_seclection))
    print(band_loction)

    time_test_start=time.clock()
    y_pr,y_real = get_oa(X_test,Y_test)
    oa=accuracy_score(y_real,y_pr)
    per_class_acc=recall_score(y_real,y_pr,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_real,y_pr)
    time_test_end=time.clock()

    num_band_seclection_now=len(band_loction)
    save_result('Indian_pines',oa,aa,kappa,num_band_seclection_now,band_loction,per_class_acc,(time_train_end-time_train_start),(time_test_end-time_test_start))

    print(per_class_acc)
    print(oa,aa,kappa)
    print((time_train_end-time_train_start),(time_test_end-time_test_start))
#    print("train accuracy %g"%acc)

    plot_max=np.zeros(np.shape(labels_ori))
    for i in range(X_train.shape[0]):
        plot_max[train_loc[0,i]][train_loc[1,i]]=labels_ori[train_loc[0,i]][train_loc[1,i]]
    for i in range(X_test.shape[0]):
        plot_max[test_loc[0,i]][test_loc[1,i]]=y_pr[i]
    
    path=os.getcwd()
    sio.savemat(path+'/plot/'+'Indian_pines'+'_'+'plot', {'plot_max':plot_max})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
