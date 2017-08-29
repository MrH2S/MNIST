# -*- coding: utf-8 -*-
'''
@brief: CNNClassifier for MNIST
        X->CONV->POOL->CONV->POOL->FLAT->DROPOUT->DESEN->LOGITS
                                                            |
                            OPTIMIZER<-LOSS<-CROSS-ENTROPY<-Y
@author: Shylock Hg
@time: 2017/8/27
@email: tcath2s@icloud.com
'''

import tensorflow as tf
import numpy as np
import csv

TRAINING_FILE='train.csv'
TEST_FILE='test.csv'

#count of traing data
NUM_TRAINING = 42000
IMG_SIZE = 28
COLOR_CHANNEL = 1
LABEL_SIZE = 10

#set log level to 'INFO'
tf.logging.set_verbosity(tf.logging.INFO)

#number to one-hot
def one_hot(num):
    label = list(range(10))
    label = [i for i in map(lambda x:0,label)]
    label[num] = 1
    return label

#load data   train_set{'images':images,'labels':labels},test_set{'images':images}
def load_training_data(file_training):
    #def load_data(training_file,test_file):
    with open(file_training) as f:
        #data set
        images = []
        labels = []
        #load csv and reformat
        reader = csv.reader(f)
        next(reader)
        for raw in reader:
            #labels
            raw_digit =[l for l in map(lambda x:int(x),raw)]
            label = one_hot(raw_digit[0])
            labels.append(label)
            #images
            image = []
            for i in range(IMG_SIZE):
                image.append([pixel for pixel in raw_digit[1+i*IMG_SIZE:1+(i+1)*IMG_SIZE]])
            images.append(image)

    return {'images':images,'labels':labels}

def load_test_data(file_test):
    with open(file_test) as f:
        #data set
        test_images = []
        #load csv and reformat
        test_reader = csv.reader(f)
        next(test_reader)
        for raw in test_reader:
            #images
            test_image = []
            test_raw_digit = [l for l in map(lambda x:int(x),raw)]
            for i in range(IMG_SIZE):
                test_image.append([pixel for pixel in test_raw_digit[i*IMG_SIZE:(i+1)*IMG_SIZE]])
            test_images.append(test_image)

    return {'images':test_images}

#load 
#test_set = load_test_data(TEST_FILE)
#training_set = load_training_data(TRAINING_FILE)


def fn_construct_model(features,labels,mode):
    #data I/O PORT
    images = tf.reshape(features['x'],(-1,IMG_SIZE,IMG_SIZE,COLOR_CHANNEL))
    #conv1
    conv1 = tf.layers.conv2d(images,32,[3,3],padding='same',activation=tf.nn.relu)
    
    #ouput [batch,14,14,32]
    pool1 = tf.layers.max_pooling2d(conv1,(2,2),2)
    
    #conv2
    conv2 = tf.layers.conv2d(pool1,64,[3,3],padding='same',activation=tf.nn.relu)
    
    #output [batch,7,7,64]
    pool2 = tf.layers.max_pooling2d(conv2,(2,2),2)
    
    flat = tf.reshape(pool2,[-1,7*7*64])
    
    #dense1
    dense1 = tf.layers.dense(flat,1024,activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(dense1,training= mode==tf.estimator.ModeKeys.TRAIN)
    
    #logits
    logits = tf.layers.dense(dropout,10)
    
    predictions = {
        'classes':tf.argmax(logits,axis=1),
        'probabilities':tf.nn.softmax(logits)
    }
    
    #predict op
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    #loss
    loss = tf.losses.softmax_cross_entropy(labels,logits)
    
    #training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        #add learning rate decay
        learning_rate = tf.train.exponential_decay(0.0001, tf.train.get_global_step(),1000, 0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        op_train = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=op_train)
    
    #eval op
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss)
 
#train
def train():
    train_set = load_training_data(TRAINING_FILE)
    #normalize
    train_set['images'] = np.asarray(train_set['images'],dtype=np.float32)/255
    train_set['labels'] = np.asarray(train_set['labels'],dtype=np.int32)
    
    #divide
    
    eval_set = {'images':None,'labels':None}
    eval_set['images'] = train_set['images'][30001:]
    eval_set['labels'] = train_set['labels'][30001:]
    
    train_set['images'] = train_set['images'][:30001]
    train_set['labels'] = train_set['labels'][:30001]
    
    
    #tensor_to_log = {'probabilities':'softmax_tensor'}
    
    #logging_hook = tf.train.LoggingTensorHook(tensor_to_log,every_n_iter=50)
    
    #train input
    train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':train_set['images']},
                                                         train_set['labels'],
                                                         num_epochs=None,
                                                         shuffle=True
                                                         )     
    
    CNNClassifier= tf.estimator.Estimator(model_fn=fn_construct_model,model_dir=r'.\tmp\model_v2')

    eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':eval_set['images']},
                                                        eval_set['labels'])
    for i in range(100):
        CNNClassifier.train(train_input_fn,steps=100)
        print(CNNClassifier.evaluate(eval_input_fn))
  
    

#predict 

def predict():
    test_set = load_test_data(TEST_FILE)
    #normalize
    test_set['images'] = np.asarray(test_set['images'],dtype=np.float32)/255
    
    CNNClassifier= tf.estimator.Estimator(model_fn=fn_construct_model,model_dir=r'.\tmp\model_v2')

    test_input_fn = tf.estimator.inputs.numpy_input_fn({'x':test_set['images']},
                                                         num_epochs=1,
                                                         shuffle=False
                                                         )
    
    predictions = CNNClassifier.predict(test_input_fn)
        
    with open(r'.\predictions.csv','w') as f:
        f.write('ImageId,Label\n')
        for i,v in enumerate(predictions):
            f.write(str(i+1)+','+str(v['classes'])+'\n')


def main(unused_argv):
    train()
   
if __name__ == '__main__':
    tf.app.run()


