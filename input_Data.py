#%%对图片使用tfrecord写入，以及进行batch化的读取

import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
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
    
    # 使用tf.decode_raw将字符串解析成图像对应的像素数组
#    images = tf.decode_raw(features['image'],tf.int32)     #VERY IMPORTANT 当是16位时使用int32解码，当是8位时使用uint8解码#
#    images = (images/65535)*255
#
    images = tf.decode_raw(features['image'],tf.uint8) 
    labels = tf.decode_raw(features['label'],tf.uint8)
    
#    images = tf.cast(features['image'],tf.int32)     #VERY IMPORTANT 当是16位时使用int32解码，当是8位时使用uint8解码#
#    labels = tf.cast(features['label'],tf.uint8)
#
    dim = raw_dim
    images = tf.reshape(images,[dim,dim,1]) 
    labels = tf.reshape(labels,[dim,dim,1])
    images = tf.image.resize_images(images, [img_W,img_H], method=0)
    labels = tf.image.resize_images(labels, [lab_W,lab_H], method=0)
    
    #在这里添加图像预处理函数（optional）
    images = images/tf.reduce_max(images)
    labels = labels/tf.reduce_max(labels)
    #使用 tf.train.batch函数来组合样例
    #这里不使用 tf.train.shuffle_batch是因为通过tf.train.string_input_producer创建文件名队列时应将乱序过了。
    #但是实验发现每次运行程序都是得到相同的mini-batch，因此考虑使用tf.train.shuffle_batch

    Image_Batch,Label_Batch = tf.train.batch([images,labels],
                                             batch_size = batch_size,
                                             num_threads = 5,                                           
                                             capacity = 100)

    return Image_Batch,Label_Batch


'''
#%%测试输入输出图片数据
import matplotlib.pyplot as plt
import numpy as np

img_W = 1024
img_H = 1024
lab_W = 512
lab_H = 512
batch_size = 3

train_images_path = 'D:\\wanghao\\dynamic_scattering\\data\\pre-processing\\'
train_labels_path = 'D:\\wanghao\\dynamic_scattering\\data\\label\\'
train_TFRecord_path = 'D:\\wanghao\\dynamic_scattering\\tfrecord_test\\train.tfrecord'

test_images_path = 'D:\\wanghao\\dynamic_scattering\\result\\pre-9-30\\'
test_labels_path = 'D:\\wanghao\\dynamic_scattering\\result\\test_label\\'
test_TFRecord_path = 'D:\\wanghao\\dynamic_scattering\\tfrecord_test\\test.tfrecord'

#generate_TFRecordfile(train_images_path,train_labels_path,train_TFRecord_path)   #调用generate_TFRecordfile函数生成TFRecord文件记录训练数据
#generate_TFRecordfile(test_images_path,test_labels_path,test_TFRecord_path)      #调用generate_TFRecordfile函数生成TFRecord文件记录测试数据

Train_Images_Batch,Train_Labels_Batch = get_batch(train_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size)
Test_Images_Batch,Test_Labels_Batch = get_batch(test_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size)

init_op =(tf.global_variables_initializer(),tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image, label= sess.run([Train_Images_Batch, Train_Labels_Batch])
    try:
        
        image, label= sess.run([Train_Images_Batch, Train_Labels_Batch])
    
        image = image[1,:,:]
        image = np.reshape(image, [img_W,img_H])
        image = Image.fromarray(image.astype('uint16'))
        
        label = label[1,:,:]
        label = np.reshape(label, [lab_W,lab_H])
        label = Image.fromarray(label.astype('uint16'))
               
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(label)
        
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
'''     
        
