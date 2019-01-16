import tensorflow as tf
import math
import time
from datetime import datetime
import pickle
import numpy as np
from tqdm import tqdm

class_num = 10
iterations = 200
image_size = 32
image_channels = 3
batch_size = 250
total_epoch = 164
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_16_logs'
model_save_path = './model/'

def _conv(input,kh,kw,n_out,dh,dw,para,name):
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope +'w',shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input,kernel,[1,dh,dw,1],padding='SAME')
        biases = tf.Variable(tf.constant(0,shape=[n_out],dtype=tf.float32), trainable=True, name='b')
        z = tf.nn.bias_add(conv,biases)
        h = tf.nn.relu(z,name=scope)
        para +=[kernel,biases]
    return h

def _fc(input,n_out,para,name):
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        W = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),trainable=True,name='b')
        h = tf.nn.relu(tf.matmul(input,W)+biases,name=scope)
        para +=[W,biases]
    return h

def _maxpool(input,kh,kw,dh,dw,name):
    return tf.nn.max_pool(input,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

def VggNet(input):
    parameters = []

    # conv_1
    conv1_1 = _conv(input,kh=3,kw=3,n_out=64,dh=1,dw=1,para=parameters,name='conv1_1')
    conv1_2 = _conv(conv1_1, kh=3, kw=3, n_out=64, dh=1, dw=1, para=parameters, name='conv1_2')
    pool1 = _maxpool(conv1_2,kh=2,kw=2,dh=2,dw=2,name='pool1')
    # conv_2
    conv2_1 = _conv(pool1,kh=3,kw=3,n_out=128,dh=1,dw=1,para=parameters,name='conv2_1')
    conv2_2 = _conv(conv2_1, kh=3, kw=3, n_out=128, dh=1, dw=1, para=parameters, name='conv2_2')
    pool2 = _maxpool(conv2_2, kh=2, kw=2, dh=2, dw=2, name='pool2')
    # conv_3
    conv3_1 = _conv(pool2, kh=3, kw=3, n_out=256, dh=1, dw=1, para=parameters, name='conv3_1')
    conv3_2 = _conv(conv3_1, kh=3, kw=3, n_out=256, dh=1, dw=1, para=parameters, name='conv3_2')
    conv3_3 = _conv(conv3_2, kh=3, kw=3, n_out=256, dh=1, dw=1, para=parameters, name='conv3_3')
    pool3 = _maxpool(conv3_3, kh=2, kw=2, dh=2, dw=2, name='pool3')
    # conv_4
    conv4_1 = _conv(pool3, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv4_1')
    conv4_2 = _conv(conv4_1, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv4_2')
    conv4_3 = _conv(conv4_2, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv4_3')
    pool4 = _maxpool(conv4_3, kh=2, kw=2, dh=2, dw=2, name='pool4')
    # conv_5
    conv5_1 = _conv(pool4, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv5_1')
    conv5_2 = _conv(conv5_1, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv5_2')
    conv5_3 = _conv(conv5_2, kh=3, kw=3, n_out=512, dh=1, dw=1, para=parameters, name='conv5_3')
    #pool5 = _maxpool(conv5_3, kh=2, kw=2, dh=2, dw=2, name='pool5')
    # FC
    flattened = tf.layers.flatten(conv5_3)
    fc1 = _fc(flattened,4096,parameters,'fc1')
    fc1_dropout = tf.nn.dropout(fc1,keep_prob=keep_prob,name='fc1_drop')
    fc2 = _fc(fc1_dropout, 4096, parameters, 'fc2')
    fc2_dropout = tf.nn.dropout(fc2, keep_prob=keep_prob, name='fc2_drop')
    fc3 = _fc(fc2_dropout, 10, parameters, 'fc3')
    softmax = tf.nn.softmax(fc3)
    return softmax

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    return data,labels


def load_data(files,data_dir,label_count):
    global image_size,image_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n,labels_n = load_data_one(data_dir+'/'+f)
        print(data_n.shape)
        data = np.append(data,data_n,axis=0)
        labels = np.append(labels,labels_n,axis=0)
    labels = np.array([[float(i==label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1,image_channels,image_size,image_size])
    data = data.transpose([0,2,3,1])
    return data,labels

def prepare_data():
    data_dir = './cifar-10-batches'
    image_dim = image_size*image_size*image_channels
    meta = unpickle(data_dir+'/batches.meta')
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' %d for d in range(1,6)]
    train_data,train_labels = load_data(train_files,data_dir,label_count)
    test_data,test_labels = load_data(['test_batch'],data_dir,label_count)
    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    return train_data,train_labels,test_data,test_labels

def data_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - np.mean(x_train[:,:,:,i])) / np.std(x_train[:,:,:,i])
        x_test[:,:,:,i] = (x_test[:,:,:,i] - np.mean(x_test[:,:,:,i])) / np.std(x_test[:,:,:,i])
    return x_train, x_test

def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001

def test(sess,epoch):
    acc = 0
    loss = 0
    pre_index = 0
    add = 1000
    for i in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index += add
        loss_,acc_ = sess.run([cross_entropy,accuracy],feed_dict={X:batch_x,Y:batch_y,keep_prob:1,train_flag:False})
        loss += loss_/10
        acc += acc_/10
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss',simple_value=loss),tf.Summary.Value(tag='test_accuracy',simple_value=acc)])
        return acc,loss,summary

if __name__=='__main__':
    train_x,train_y,test_x,test_y = prepare_data()
    train_x,test_x = data_preprocessing(train_x,test_x)
    X = tf.placeholder(dtype=tf.float32,shape=[None,image_size,image_size,3])
    Y = tf.placeholder(dtype=tf.float32,shape=[None,class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    output = VggNet(X)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits = output))
    train_step = tf.train.MomentumOptimizer(learning_rate,momentum_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)

        for epoch in tqdm(range(1,total_epoch+1)):
            lr = learning_rate_schedule(epoch)
            pre_index = 0
            train_acc = 0
            train_loss = 0
            start_time = time.time()


            for i in range(1,iterations+1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]
                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={X:batch_x,Y:batch_y,keep_prob:1,train_flag:True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if i==iterations:
                    train_loss /=iterations
                    train_acc /= iterations

                    loss,acc = sess.run([cross_entropy,accuracy],feed_dict={X:batch_x,Y:batch_y,keep_prob:1,train_flag:True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy',simple_value=train_acc),tf.Summary.Value(tag='train_accuracy',simple_value=train_acc)])

                    val_acc, val_loss, test_summary = test(sess,epoch)

                    summary_writer.add_summary(train_summary,epoch)
                    summary_writer.add_summary(test_summary, epoch)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (i, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                          % (i, iterations, train_loss / i, train_acc / i))

        save_path = saver.save(sess,model_save_path)
        print("Model saved in file: %s" % save_path)
