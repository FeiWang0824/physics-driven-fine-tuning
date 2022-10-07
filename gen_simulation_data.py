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

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2
from model_Unet_GIDC_wDGI import DGI_reconstruction, image_cut_by_std    

###########################
image_name = 'stl10' # name of your image used for simulation 
dataSet = 'stl10'    # 'CelebA' or 'stl10'
dim = 64             # 128 for dataSet = 'stl10'
                      # 64 for dataSet = 'CelebA'
############################

num_patterns = 1024  
NN = 'Unet'
mode = 'wDGI'

# learned patterns
pattern_save_path = '.\\model\\trained_%s_patterns_%d_%s_%s_%d.mat'%(dataSet,num_patterns,NN,mode,dim)
trained_patterns = sio.loadmat(pattern_save_path)
trained_patterns = trained_patterns['trained_patterns']

# random patterns
random_patterns = np.random.randn(dim,dim,1,num_patterns).astype(np.float32)
random_patterns[random_patterns > 0] = 1
random_patterns[random_patterns < 0] = 0

# test image
im = cv2.imread('.\\data\\images\\%s.bmp'%image_name)
im = cv2.resize(im, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
im = im[:,:,0]
im = (im - np.min(im)) / (np.max(im) - np.min(im))

im_tf = tf.constant(im)
im_tf = tf.reshape(im_tf, [1,dim,dim,1])

# measurement process
# learned patterns
y = tf.nn.conv2d(im_tf,trained_patterns,strides=[1,1,1,1],padding='VALID')  
mean_y, variance_y = tf.nn.moments(y, [0,1,2,3])
y = (y - mean_y)/tf.sqrt(variance_y)
y_learn = tf.cast(y, tf.float32)

# random patterns
y = tf.nn.conv2d(im_tf,random_patterns,strides=[1,1,1,1],padding='VALID')  
mean_y, variance_y = tf.nn.moments(y, [0,1,2,3])
y = (y - mean_y)/tf.sqrt(variance_y)
y_rand = tf.cast(y, tf.float32)

# DGI reconstruction
dgi_learn = DGI_reconstruction(y_learn,trained_patterns,num_patterns,dim,dim,0.5)
dgi_rand = DGI_reconstruction(y_rand,random_patterns,num_patterns,dim,dim,0.5)

with tf.Session() as sess:
    y_learn_s = sess.run(y_learn).reshape([num_patterns,1])
    dgi_learn_s = sess.run(dgi_learn).reshape([dim, dim])
    
    y_rand_s = sess.run(y_rand).reshape([num_patterns,1])
    dgi_rand_s = sess.run(dgi_rand).reshape([dim, dim])
    
    sio.savemat('.\\data\\' + image_name + '_sim.mat', {'y':y_learn_s, 'dgi_r':dgi_learn_s, 'GT':im})
    
    dgi_learn_post = image_cut_by_std(dgi_learn_s, 2)  # post processing trick for better visualization
    dgi_rand_post = image_cut_by_std(dgi_rand_s, 2)    # post processing trick for better visualization
    
    plt.subplot(231)
    plt.imshow(random_patterns[:,:,0,0])
    plt.title('rand pattern')
    plt.axis('off')
    plt.subplot(232)
    plt.plot(y_rand_s)
    plt.title('measurments')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(dgi_rand_post)
    plt.title('DGI-rand')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(trained_patterns[:,:,0,0])
    plt.axis('off')
    plt.title('learned pattern')
    plt.subplot(235)
    plt.plot(y_learn_s)
    plt.title('measurments')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(dgi_learn_post)
    plt.title('DGI-learn')
    plt.axis('off')

    























