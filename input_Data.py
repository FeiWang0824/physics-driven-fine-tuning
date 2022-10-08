#%%对图片使用tfrecord写入，以及进行batch化的读取

import os
import tensorflow as tf
from PIL import Image 
import numpy as np

# 生成字符串型的属性
def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_TFRecordfile(image_path,label_path,TFRecord_path,num_min,num_max):
    images = []
    labels = []
    # 获取每一个样本的路径
    for file in os.listdir(image_path):
        images.append(image_path+file)
    for file in os.listdir(label_path):
        labels.append(label_path+file)
    num_examples = len(images)
    
    print('There are %d images \n'%(num_examples))
    print('%d images were used! \n'%(num_max-num_min+1))
    
    writer = tf.python_io.TFRecordWriter(TFRecord_path)#创建一个writer写TFRecord文件
    for index in range(num_min,num_max):
        image = Image.open(images[index])
        image = image.tobytes()
        label = Image.open(labels[index])
        label = label.tobytes()
        
        #将一个样例转换为Example Protocol Buffer的格式，并且将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image':_bytes_feature(image),
            'label':_bytes_feature(label)}))
        
        writer.write(example.SerializeToString())   #将一个Example 写入TFRecord文件
    writer.close()

def get_batch(TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size,raw_dim):
    
    reader = tf.TFRecordReader() # 创建一个reader来读取TFRecord文件中的样例 
    
    files = tf.train.match_filenames_once(TFRecord_path) # 获取文件列表
    filename_queue = tf.train.string_input_producer(files,shuffle = False,num_epochs = None) # 创建文件名队列，乱序，每个样本使用num_epochs次
    
    # 读取并解析一个样本
    _,example = reader.read(filename_queue)
    features = tf.parse_single_example(
        example,
        features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.string)})
    
    images = tf.decode_raw(features['image'],tf.uint8) 
    labels = tf.decode_raw(features['label'],tf.uint8)
    
    dim = raw_dim
    images = tf.reshape(images,[dim,dim,1]) 
    labels = tf.reshape(labels,[dim,dim,1])
    images = tf.image.resize_images(images, [img_W,img_H], method=0)
    labels = tf.image.resize_images(labels, [lab_W,lab_H], method=0)
    
    #在这里添加图像预处理函数（optional）
    images = images/tf.reduce_max(images)
    labels = labels/tf.reduce_max(labels)

    Image_Batch,Label_Batch = tf.train.batch([images,labels],
                                             batch_size = batch_size,
                                             num_threads = 5,                                           
                                             capacity = 100)

    return Image_Batch,Label_Batch


