import tensorflow as tf
import numpy as np

def GroupNorm(x,name_scope,group=8,esp=1e-5):
    with tf.variable_scope(name_scope):
        x = tf.transpose(x,[0,4,1,2,3])
        N,C,H,W,D = x.get_shape().as_list()
        G = min(group,C)
        x = tf.reshape(x, [-1,G,C//G,H,W,D])
        mean, var = tf.nn.moments(x,[2,3,4,5])
        mean = tf.reshape(mean,[-1,G,1,1,1,1])
        var = tf.reshape(var,[-1,G,1,1,1,1])
        x = (x - mean) / tf.sqrt(var + esp)
        gamma = tf.get_variable('gamma',[C],initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta',[C],initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1,C,1,1,1])
        beta = tf.reshape(beta, [1,C,1,1,1])
        output = tf.reshape(x, [-1,C,H,W,D]) * gamma + beta
        output = tf.transpose(output, [0,2,3,4,1])  # [bs,h,w,d,c]
    return output

def ResidualBlock(x,filters,stride,name_scope):
    with tf.variable_scope(name_scope):
        N,H,W,D,C = x.get_shape().as_list()
        residual=x
        out = tf.layers.conv3d(x,filters,kernel_size=(3,3,3),strides=stride,padding='same',name='conv1')
        out = GroupNorm(out,name_scope='gn1',group=8)
        out = tf.nn.leaky_relu(out)
        out = tf.layers.conv3d(out,filters,kernel_size=(3,3,3),strides=1,padding='same',name='conv2')
        out = GroupNorm(out,name_scope='gn2',group=8)
        if C!=filters or stride==2:
            y = tf.layers.conv3d(x,filters,kernel_size=(3,3,3),strides=stride,padding='same',name='conv3')
            residual=GroupNorm(y,name_scope='gn3',group=8)
        out+=residual
        out=tf.nn.leaky_relu(out)
        return out

def ResNet34(x,num_classes,name_scope):
    with tf.variable_scope(name_scope):
        num_channels=[16,32,64,128]
        #input (bs,128,128,32,4)
        #stage1
        x=tf.layers.conv3d(x,num_channels[0],kernel_size=(3,3,3),strides=1,padding='same',name='conv1') #(bs,128,128,32,16)
        x=GroupNorm(x,name_scope='gn1',group=8)   
        x=tf.nn.leaky_relu(x)
        x=tf.layers.conv3d(x,num_channels[0],kernel_size=(3,3,3),strides=2,padding='same',name='conv2') #(bs,64,64,16,16)
        x=GroupNorm(x,name_scope='gn2',group=8)   
        x=tf.nn.leaky_relu(x)
        x=ResidualBlock(x,num_channels[0],stride=1,name_scope='block1')  #(bs,64,64,16,16)
        x=ResidualBlock(x,num_channels[0],stride=1,name_scope='block2')  #(bs,64,64,16,16)
        x=ResidualBlock(x,num_channels[0],stride=1,name_scope='block3')  #(bs,64,64,16,16)
        #stage2
        x=ResidualBlock(x,num_channels[1],stride=2,name_scope='block4')  #(bs,32,32,8,32)
        x=ResidualBlock(x,num_channels[1],stride=1,name_scope='block5')  #(bs,32,32,8,32)
        x=ResidualBlock(x,num_channels[1],stride=1,name_scope='block6')  #(bs,32,32,8,32)
        x=ResidualBlock(x,num_channels[1],stride=1,name_scope='block7')  #(bs,32,32,8,32)
        #stage3
        x=ResidualBlock(x,num_channels[2],stride=2,name_scope='block8')  #(bs,16,16,4,64)
        x=ResidualBlock(x,num_channels[2],stride=1,name_scope='block9')  #(bs,16,16,4,64)
        x=ResidualBlock(x,num_channels[2],stride=1,name_scope='block10')  #(bs,16,16,4,64)
        x=ResidualBlock(x,num_channels[2],stride=1,name_scope='block11')  #(bs,16,16,4,64)
        x=ResidualBlock(x,num_channels[2],stride=1,name_scope='block12')  #(bs,16,16,4,64)
        x=ResidualBlock(x,num_channels[2],stride=1,name_scope='block13')  #(bs,16,16,4,64)
        #stage4
        x=ResidualBlock(x,num_channels[3],stride=2,name_scope='block14')  #(bs,8,8,2,128)
        x=ResidualBlock(x,num_channels[3],stride=1,name_scope='block15')  #(bs,8,8,2,128)
        x=ResidualBlock(x,num_channels[3],stride=1,name_scope='block16')  #(bs,8,8,2,128)
        #stage5
        _,H,W,D,C = x.get_shape().as_list()
        x=tf.layers.average_pooling3d(x,pool_size=(H,W,D),strides=1)  
        x=tf.reshape(x,[-1,C])      #(bs,128)
        #x=tf.layers.dense(x,units=64,activation=tf.nn.sigmoid,name='l1')   #(bs,64)
        out=tf.layers.dense(x,units=num_classes,activation=tf.nn.softmax,name='l2')
        return out
    