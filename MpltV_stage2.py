# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:24 2019

@author: HP
"""

import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import glob
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
import cv2
import numpy as np
import pandas as pd
import keras as kr
from keras.models import load_model
from keras.layers import Activation, Dense

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from skimage import io,data
import time
#from keras import layers
#from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow import keras 
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 


    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
#        # train_acc        
        plt.plot(iters, self.accuracy[loss_type], 'coral', label='train acc', linewidth=2)     
        # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'dodgerblue', label='val acc', linewidth=2) 

##        y = range(0,1.2,0.2)
#        y = np.arange(0.6,1,0.1)
#        plt.yticks(y) 
           
        plt.grid(axis="y")
        plt.xlabel(loss_type)
        plt.ylabel('accuracy')
        plt.legend(loc="best")    
        
        plt.savefig('fig_acc.png')            
#    if loss_type == 'epoch':
        plt.figure()       
        # val_acc
        plt.plot(iters, self.losses[loss_type], 'coral', label='train loss', linewidth=2)
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'dodgerblue', label='val loss', linewidth=2)
        plt.grid(axis="y")
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="best")
        
#        y2 = np.arange(0,1.2,0.2)
#        plt.yticks(y2) 

        plt.savefig('fig_loss.png')        
#        plt.show()





import os,sys
os.getcwd()
os.chdir("/home/cjd/30_Crop_train")
print(os.getcwd())
print (sys.version)


now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#import os
# 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
kr.backend.tensorflow_backend.set_session(tf.Session(config=config))


import tensorflow as tf        
def focal_loss(gamma=2.):            
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  



batch_size = 64 
epochs = 30
MODEL_INIT = './obj_reco/init_model.h5'
MODEL_PATH = './obj_reco/tst_model.h5'
board_name1 = './obj_reco/stage1/' + now + '/'
board_name2 = './obj_reco/stage2/' + now + '/'

train_dir1='/home/wkq/Projects/kerasVGG19/train_b/'
validation_dir1='/home/wkq/Projects/kerasVGG19/test_b/'

train_dir2='/home/cjd/22_2FCS_rev/train_seg/'
validation_dir2='/home/cjd/22_2FCS_rev/test_seg/'

img_size = (224, 224)  
#classes=list(range(1,5))
#classes=['1','2','3','4']

nb_train_samples1 = len(glob.glob(train_dir1 + '/*/*.*'))  
nb_validation_samples1 = len(glob.glob(validation_dir1 + '/*/*.*'))  
classes1 = sorted([o for o in os.listdir(train_dir1)])  

nb_train_samples2 = len(glob.glob(train_dir2 + '/*/*.*'))  
nb_validation_samples2 = len(glob.glob(validation_dir2 + '/*/*.*'))  
classes2 = sorted([o for o in os.listdir(train_dir2)])  




#--------------MS-DenseNet---------------------------------------
class SeBlock(keras.layers.Layer):   
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction
    def build(self,input_shape):
    	#input_shape     
    	pass
    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(int(x.shape[-1]) // self.reduction, use_bias=False,activation=keras.activations.relu)(x)
        x = keras.layers.Dense(int(inputs.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(x)
        return keras.layers.Multiply()([inputs,x])    
        #return inputs*x 

def get_bottlenet(image_size,alpha=1.0):
    inputs = keras.layers.Input(shape=(image_size,image_size,3),name='input_1')
    net = keras.layers.ZeroPadding2D(padding=(3,3),name='zero_padding2d_1')(inputs)
    net = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2),
                             padding='valid', name='conv1/conv')(net)
    net = keras.layers.BatchNormalization(name='conv1/bn')(net)
    net = keras.layers.ReLU(name='conv1/relu')(net)
    net = keras.layers.ZeroPadding2D(padding=(1,1),name='zero_padding2d_2')(net)
#    se = keras.layers.GlobalAveragePooling2D(name='transform2_pool')(net)
    se=SeBlock()(net)

    net = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',
                                   name='pool1')(net)
#    se = keras.layers.GlobalAveragePooling2D(name='transform2_pool')(net)
    se=SeBlock()(net)
    
    for i in range(int(3*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32,kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv2_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv2_block{}_concat'.format(i))([net,block])    
    net = keras.layers.BatchNormalization(name='pool2_bn')(net) 
    net = keras.layers.ReLU(name='pool2_relu')(net)
    eq = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform2_dense1')(se)
    net = keras.layers.Multiply(name='transform2_multiply')([net,eq])

    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                              padding='same',name='pool2_conv')(net)
    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                        name='pool2_pool')(net)
#    se = keras.layers.GlobalAveragePooling2D(name='transform3_pool')(net)
    se=SeBlock()(net)

    for i in range(int(6*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1),
                                             padding='same',
                                             name='conv3_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv3_block{}_concat'.format(i))([net,block])
    net = keras.layers.BatchNormalization(name='pool3_bn')(net) 
    net = keras.layers.ReLU(name='pool3_relu')(net)
    eq = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform3_dense1')(se)
    net = keras.layers.Multiply(name='transform3_multiply')([net,eq])

    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                              padding='same',name='pool3_conv')(net)
    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                    name='pool3_pool')(net)
#    se = keras.layers.GlobalAveragePooling2D(name='transform4_pool')(net)
    se=SeBlock()(net)
    for i in range(int(12*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv4_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv4_block{}_concat'.format(i))([net,block])
    net = keras.layers.BatchNormalization(name='pool4_bn')(net) 
    net = keras.layers.ReLU(name='pool4_relu')(net)
    eq = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform4_dense1')(se)
    net = keras.layers.Multiply(name='transform4_multiply')([net,eq])

    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                             padding='same',name='pool4_conv')(net)
    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                        name='pool4_pool')(net)
#    se = keras.layers.GlobalAveragePooling2D(name='transform5_pool')(net)
    se=SeBlock()(net)
    for i in range(int(8*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv5_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv5_block{}_concat'.format(i))([net,block])
       
    eq = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform5_dense1')(se)
    net = keras.layers.Multiply(name='transform5_multiply')([net,eq])
    net = keras.layers.BatchNormalization(name='bn')(net)
    net = keras.layers.ReLU(name='relu')(net)
    model = keras.Model(inputs=inputs,outputs=net,name='mobile_densenet_bottle')
    return model

def get_model1(image_size=224,alpha=1.0,classes=10):
    bottlenet = get_bottlenet(alpha=alpha,image_size=image_size)
    net = keras.layers.GlobalAveragePooling2D(name='global_pool')(bottlenet.output)
    net = keras.layers.Dropout(rate=0.4,name='dropout1')(net)
    net = keras.layers.Dropout(rate=0.4,name='dropout2')(net)
    output = keras.layers.Dense(units=classes,activation='softmax',
                             name='prediction1')(net)
    model = keras.Model(inputs=bottlenet.input,outputs=output,name='mobile_densenet')
    return model

def get_model2(image_size=224,alpha=1.0,classes=10):
    bottlenet = get_bottlenet(alpha=alpha,image_size=image_size)
    net = keras.layers.GlobalAveragePooling2D(name='global_pool')(bottlenet.output)
    net = keras.layers.Dropout(rate=0.4,name='dropout1')(net)
    net = keras.layers.Dropout(rate=0.4,name='dropout2')(net)
    output = keras.layers.Dense(units=classes,activation='softmax',
                             name='prediction2')(net)
    model = keras.Model(inputs=bottlenet.input,outputs=output,name='mobile_densenet')
    return model

model1 = get_model1(image_size=224,classes=len(classes1))
model2 = get_model2(image_size=224,classes=len(classes2))
#model.summary()


'''
#------#PeleeNet------------------------
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def conv_bn_relu(input_tensor, ch, kernel, padding="same", strides=1, weight_decay=5e-4):
    x = layers.Conv2D(ch, kernel, padding=padding, strides=strides,
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def stem_block(input_tensor):
    x = conv_bn_relu(input_tensor, 32, 3, strides=2)
    branch1 = conv_bn_relu(x, 16, 1)
    branch1 = conv_bn_relu(branch1, 32, 3, strides=2)
    branch2 = layers.MaxPool2D(2)(x)
    x = layers.Concatenate()([branch1, branch2])
    return conv_bn_relu(x, 32, 1)

def dense_block(input_tensor, num_layers, growth_rate, bottleneck_width):
    x = input_tensor
    growth_rate = int(growth_rate / 2)

    for i in range(num_layers):
        inter_channel = int(growth_rate*bottleneck_width/4) * 4
        branch1 = conv_bn_relu(x, inter_channel, 1)
        branch1 = conv_bn_relu(branch1, growth_rate, 3)

        branch2 = conv_bn_relu(x, inter_channel, 1)
        branch2 = conv_bn_relu(branch2, growth_rate, 3)
        branch2 = conv_bn_relu(branch2, growth_rate, 3)
        x = layers.Concatenate()([x, branch1, branch2])
    return x

def transition_layer(input_tensor, k, use_pooling=True):
    x = conv_bn_relu(input_tensor, k, 1)
    if use_pooling:
        return layers.AveragePooling2D(2)(x)
    else:
        return x

def PeleeNet1(input_shape=(224,224,3), use_stem_block=True, n_classes=1000):
    n_dense_layers = [3,4,8,6]
    bottleneck_width = [1,2,4,4]
    out_layers = [128,256,512,704]
    growth_rate = 32

    input = layers.Input(input_shape)
    x = stem_block(input) if use_stem_block else input
    for i in range(4):
        x = dense_block(x, n_dense_layers[i], growth_rate, bottleneck_width[i])
        use_pooling = i < 3
        x = transition_layer(x, out_layers[i], use_pooling=use_pooling)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax", name='class1')(x)
    return keras.models.Model(input, x)


def PeleeNet2(input_shape=(224,224,3), use_stem_block=True, n_classes=1000):
    n_dense_layers = [3,4,8,6]
    bottleneck_width = [1,2,4,4]
    out_layers = [128,256,512,704]
    growth_rate = 32

    input = layers.Input(input_shape)
    x = stem_block(input) if use_stem_block else input
    for i in range(4):
        x = dense_block(x, n_dense_layers[i], growth_rate, bottleneck_width[i])
        use_pooling = i < 3
        x = transition_layer(x, out_layers[i], use_pooling=use_pooling)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax", name='class2')(x)
    return keras.models.Model(input, x)


#model = PeleeNet(input_shape=(224,224,3),n_classes=len(classes))
#model.summary()

model1 = PeleeNet1(input_shape=(224,224,3),n_classes=len(classes1))
model2 = PeleeNet2(input_shape=(224,224,3),n_classes=len(classes2))
'''


'''
#---------Efficientnet transfer learning--------------------------------------------------------------
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import regularizers
import efficientnet.keras as efn 


IMG_SHAPE=(224, 224, 3)


base_model = efn.EfficientNetB0(input_shape=IMG_SHAPE,include_top=False, 
weights='/home/cjd/01_rice_dete/obj_reco/checkpoint/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')


for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics = ['accuracy'])  #rmsprop
'''


train_datagen = ImageDataGenerator(validation_split=0.2)
train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))  
train_data1 = train_datagen.flow_from_directory(train_dir1, target_size=img_size, classes=classes1)
train_data2 = train_datagen.flow_from_directory(train_dir2, target_size=img_size, classes=classes2)

validation_datagen = ImageDataGenerator()
validation_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
validation_data1 = validation_datagen.flow_from_directory(validation_dir1, target_size=img_size, classes=classes1)
validation_data2 = validation_datagen.flow_from_directory(validation_dir2, target_size=img_size, classes=classes2)
'''
# -------------1st stage-----------------
#model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, save_best_only=True, monitor='val_accuracy', mode='max')
model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, monitor='val_accuracy')
board1 = TensorBoard(log_dir=board_name1,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list1 = [model_checkpoint1, board1]

model1.compile(optimizer=keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'])

model1.fit_generator(train_data1, steps_per_epoch=nb_train_samples1 / float(batch_size),
                           epochs = epochs,
                           validation_steps=nb_validation_samples1 / float(batch_size),
                           validation_data=validation_data1,
                           callbacks=callback_list1, verbose=2)
'''
#---------------2n stage---------------------------------------------
model_checkpoint2 = ModelCheckpoint(filepath=MODEL_PATH,  monitor='val_accuracy')
board2 = TensorBoard(log_dir=board_name2,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list2 = [model_checkpoint2, board2]


model2.load_weights(MODEL_INIT,by_name=True)
for lyrs in model2.layers:
    lyrs.trainable = True


'''
#-----tensorflow keras -----------------------
learning_rate = 0.01
#learning_rate = 0.0001
decay = 1e-6
momentum = 0.8
nesterov = True
sgd_optimizer = keras.optimizers.SGD(lr = learning_rate, decay = decay,            
                    momentum = momentum, nesterov = nesterov)
model2.compile(loss = [focal_loss(gamma=2)],
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])
'''

model2.compile(optimizer=keras.optimizers.Adam(), loss =[focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy', Adadelta
#model.compile(optimizer=optimizers.SGD(lr=0.0001), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',
##model.compile(optimizer=optimizers.Adadelta(), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',


history = LossHistory()

model2.fit_generator(train_data2,steps_per_epoch=nb_train_samples2/float(batch_size),epochs=epochs,
                    validation_data=validation_data2,validation_steps=nb_validation_samples2/float(batch_size),
#                    callbacks=callback_list2,
                    callbacks=[history],                    
                    verbose=2)


model2.save(MODEL_PATH)
history.loss_plot('epoch')



