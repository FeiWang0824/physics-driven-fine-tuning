# -*- coding: utf-8 -*-
"""
By Fei Wang, May 6, 2021
Contact: WangFei_m@outlook.com
This code implements the model-driven fine tune process
reported in the paper: 
Fei Wang, Chenglong Wang, Chenjin Deng, Shensheng Han, and Guohai Situ. 'Single-pixel imaging using physics enhanced deep learning,'  
Please cite our paper if you find this code offers any help.

Inputs:
DGI: dim x dim         : DGI results
y:   1 x num_patterns  : raw measurements
trained_patterns: dim x dim x num_patterns : learned sampling patterns

Outputs:
DLDC_r: dim x dim x steps
steps=0 is actually the physics-informed (DGI) DL results (data-driven)  
others are results of physics-driven fine tuning process
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model_Unet_GIDC_wDGI
import scipy.io as sio
import os

np.random.seed(1)

data_name = 'stl10_sim'
dataSet = 'stl10'  

lr0 = 0.001
steps = 300
print_freq = 100
num_patterns = 1024 

data_mode = data_name.split('_')[-1].lower()
assert (data_mode in {'sim', 'exp'})
isExp = True if data_mode == 'exp' else False
                                                                                                                                
mode = 'wDGI'
NN = 'Unet'
batch_size = 1

# raw measurements
data_save_path = '.\\data\\%s.mat'%(data_name)
data = sio.loadmat(data_save_path)

y_raw = data['y']
DGI = data['dgi_r']
GT = data['GT']   
dim = DGI.shape[0]
img_W = dim
img_H = dim
lab_W = dim
lab_H = dim

model_save_path = '.\\model\\model_%s_%d_%s_%s_%d.ckpt'%(dataSet,num_patterns,NN,mode,dim)
pattern_save_path = '.\\model\\trained_%s_patterns_%d_%s_%s_%d.mat'%(dataSet,num_patterns,NN,mode,dim)
result_save_path = '.\\results\\%s_r.mat'%(data_name)


if not os.path.exists('.\\results\\'):
    os.makedirs('.\\results\\')

# learned patterns
trained_patterns = sio.loadmat(pattern_save_path)
trained_patterns = trained_patterns['trained_patterns']

DLDC_r = np.zeros([dim,dim,steps])   

tf.reset_default_graph()       
# input placeholder
with tf.variable_scope('input'):
    inpt = tf.placeholder(tf.float32, shape=[batch_size,img_W,img_H,1],name = 'DGI-inpt') 
    x = tf.placeholder(tf.float32, shape=[batch_size,img_W,img_H,1],name = 'label') 
    y = tf.placeholder(tf.float32,shape=[batch_size,num_patterns],name = 'y') 
    A = tf.placeholder(tf.float32,shape=[img_W,img_H,1,num_patterns],name = 'A')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    
# forward propagation of DNN and SPI image formation
x_out,y_out = model_Unet_GIDC_wDGI.inference(inpt, A, img_W, img_H, batch_size, num_patterns, isExp)

# loss function
measure_loss = tf.losses.mean_squared_error(y, y_out)
    
# loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv')
loss_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv1')
loss_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv2')
loss_vars3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv3')     
loss_vars = [loss_vars1,loss_vars2,loss_vars3]
    
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(measure_loss,var_list=loss_vars)
    
init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())
saver = tf.train.Saver()
    
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess,model_save_path)    
    y_temp = np.reshape(y_raw,[1,num_patterns])
            
    DGI = DGI/np.max(DGI)#(DGI - np.min(DGI))/(np.max(DGI) - np.min(DGI))
    val_images_batch = np.reshape(DGI, [batch_size, img_W, img_H, 1])      
    DGI = model_Unet_GIDC_wDGI.image_cut_by_std(DGI,2)
    
    print('starting model-based Fine tune process...')        
    # fine tune
    for step in range(steps):
        lr_temp = lr0
    
        DLDC_r[:, :, step] = sess.run(x_out, feed_dict={inpt: val_images_batch, y: y_temp, A: trained_patterns}).reshape([img_W, img_H])
    
        loss_measure = sess.run(measure_loss, feed_dict={inpt: val_images_batch, y: y_temp, A: trained_patterns, lr: lr_temp})
    
        if step == 0:
            x_pred0 = sess.run(x_out, feed_dict={inpt: val_images_batch, y: y_temp, A: trained_patterns, lr: lr_temp})
            x_pred0 = np.reshape(x_pred0[0, :, :, :], [img_W, img_H])
    
        if step % print_freq == 0 or step == steps-1:
            x_pred, y_pred, loss_measure = sess.run([x_out, y_out, measure_loss],
                                                    feed_dict={inpt: val_images_batch, y: y_temp, A: trained_patterns, lr: lr_temp})
    
            x_pred = np.reshape(x_pred[0, :, :, :], [img_W, img_H])
            y_pred = np.reshape(y_pred[0, :], [1, num_patterns])
    
            plt.subplot(231)
            plt.plot(np.transpose(y_temp[0, :]))
            plt.title('raw')
            plt.axis('off')
            plt.subplot(232)
            plt.plot(np.transpose(y_pred))
            plt.title('reproduced')
            plt.axis('off')
            plt.subplot(233)
            plt.imshow(DGI)
            plt.title('DGI-learned')
            plt.axis('off')
            plt.subplot(234)
            plt.imshow(x_pred0)
            plt.title('Informed')
            plt.axis('off')
            plt.subplot(235)
            plt.imshow(x_pred)
            plt.title('Fine-tune:%d'%step)
            plt.axis('off')
            plt.subplot(236)
            plt.imshow(GT)
            plt.title('Ground truth')
            plt.axis('off')
            plt.show()
    
            print('[step: %d] --- measure loss: %f' % (step, loss_measure))
    
        sess.run(train_op, feed_dict={inpt: val_images_batch, y: y_temp, A: trained_patterns, lr: lr_temp})
                                
sio.savemat(result_save_path,{'im_pred':DLDC_r})

        
        