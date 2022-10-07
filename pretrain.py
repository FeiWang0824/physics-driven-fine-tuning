# -*- coding: utf-8 -*-
"""
By Fei Wang, May 6, 2021
Contact: WangFei_m@outlook.com
This code implements the DNN model training
reported in the paper: 
Fei Wang, Chenglong Wang, Chenjin Deng, Shensheng Han, and Guohai Situ. 'Single-pixel imaging using physics enhanced deep learning,'  
Please cite our paper if you find this code offers any help.

Inputs:
training dataset 

Outputs:
trained model
learned patterns
learning curve
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_Data
import model_Unet_GIDL_wDGI
import scipy.io as sio
import os

tf.reset_default_graph()

np.random.seed(1)
batch_size = 30
raw_dim = 256 # 获取图片的原始尺寸
dim = 128     # 实际用于训练的图片尺寸
img_W = dim
img_H = dim
lab_W = dim
lab_H = dim
num_patterns = 1024
epoch = 64
lr = 0.0002
num_train_imgs = 29000

dataSet = 'CelebA'
NN = 'Unet'
mode = 'wDGI'

step_num = int(epoch*num_train_imgs/batch_size + 1)
epoch_step = int(num_train_imgs/batch_size)

model_save_path = '.\\model\\model_%s_%d_%s_%s_%d.ckpt'%(dataSet,num_patterns,NN,mode,dim)
pattern_save_path = '.\\model\\trained_%s_patterns_%d_%s_%s_%d.mat'%(dataSet,num_patterns,NN,mode,dim)
loss_save_path = '.\\model\\loss_%s_%s_%s_%d_%d.mat'%(dataSet,NN,mode,num_patterns,dim)

# data
train_TFRecord_path = '.\\data\\tfrecord\\train_%s.tfrecord'%(dataSet)
val_TFRecord_path = '.\\data\\tfrecord\\val_%s.tfrecord'%(dataSet)
input_data_path = '.\\data\\%s\\'%(dataSet)  # 存储图片的位置 autoencoder 输入和标签相同
label_data_path = '.\\data\\%s\\'%(dataSet)  

print('Please wait for generating TFRecord files ...')
# Generating .tfrecord
if not os.path.exists(train_TFRecord_path):
    input_Data.generate_TFRecordfile(input_data_path,label_data_path,train_TFRecord_path,0,28999) #调用generate_TFRecordfile函数生成TFRecord文件
if not os.path.exists(val_TFRecord_path):
    input_Data.generate_TFRecordfile(input_data_path,label_data_path,val_TFRecord_path,29000,29999)
print('Finished')

Train_Images_Batch,Train_Labels_Batch = input_Data.get_batch(train_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size,raw_dim)
Val_Images_Batch,Val_Labels_Batch = input_Data.get_batch(val_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size,raw_dim)

# input placeholder
with tf.variable_scope('input'):
    inpt = tf.placeholder(tf.float32, shape=[batch_size,img_W,img_H,1],name = 'images')
    label = tf.placeholder(tf.float32,shape=[batch_size,lab_W,lab_H,1],name = 'labels')

DGI_r,patterns,x_out = model_Unet_GIDL_wDGI.inference(inpt, img_W, img_H, batch_size, num_patterns)
         
# loss function
loss = tf.losses.mean_squared_error(x_out, inpt)
# loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='weights')

# train
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
        
init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    # saver.restore(sess,model_save_path)
    coord = tf.train.Coordinator() # 线程终止(should_stop=True,request_stop)
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) # 启动多个线程操作同一队列  
    print('Start training on %s'%(dataSet))
    
    count = 0
    train_loss = np.zeros(epoch+1)
    val_loss = np.zeros(epoch+1)
    try:
        for step in range(step_num):
            if coord.should_stop():
                print('check tfrecord data!')                
                break
                        
            train_images_batch,train_labels_batch = sess.run([Train_Images_Batch,Train_Labels_Batch])
            train_images_batch = np.reshape(train_images_batch,[batch_size,img_W,img_H,1])
            train_labels_batch = np.reshape(train_labels_batch,[batch_size,lab_W,lab_H,1])
            
            if  step%epoch_step == 0:
                val_images_batch,val_labels_batch = sess.run([Val_Images_Batch,Val_Labels_Batch])
                val_images_batch = np.reshape(val_images_batch,[batch_size,img_W,img_H,1])
                val_labels_batch = np.reshape(val_labels_batch,[batch_size,lab_W,lab_H,1])

                train_loss[count] = sess.run(loss,feed_dict={inpt:train_images_batch,label:train_labels_batch})
                val_loss[count] = sess.run(loss,feed_dict={inpt:val_images_batch,label:val_labels_batch})
                                
                print('[epoch %d]: loss on training batch:%f  loss on validation batch:%f' % (count,train_loss[count],val_loss[count]))
                count = count + 1
                x_pred = sess.run(x_out,feed_dict={inpt:val_images_batch,label:val_labels_batch})
                x_pred = np.reshape(x_pred[0,:,:,:],[img_W,img_H]) 
                A = sess.run(patterns)
                A_10 = np.reshape(A[:,:,:,10],[img_W,img_H])
                Obj = np.reshape(val_labels_batch[0,:,:,:],[img_W,img_H]) 
                
                DGI_pred = sess.run(DGI_r,feed_dict={inpt:val_images_batch,label:val_labels_batch})
                DGI_pred = np.reshape(DGI_pred[0,:,:,:],[img_W,img_H])
                    
                plt.subplot(141)
                plt.imshow(A_10)
                plt.title('learned Pattern')     
                plt.axis('off')
                plt.subplot(142)
                plt.imshow(DGI_pred) 
                plt.title('DGI pred.')  
                plt.axis('off')
                plt.subplot(143)
                plt.imshow(x_pred)
                plt.title('DL pred.')   
                plt.axis('off')
                plt.subplot(144)
                plt.imshow(Obj) 
                plt.title('Ground truth')                
                plt.subplots_adjust(hspace=0.25, wspace=0.25)
                plt.axis('off')
                plt.show()    
                                        
            sess.run(train_op,feed_dict={inpt:train_images_batch,label:train_labels_batch})
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        coord.request_stop()
    finally:
        coord.request_stop()
    coord.join(threads)
    
    saver.save(sess, model_save_path)
    sio.savemat(pattern_save_path,{'trained_patterns':A})
    sio.savemat(loss_save_path,{'train_loss':train_loss,'val_loss':val_loss})

            

        




















    
    
    
    
    
    
    
