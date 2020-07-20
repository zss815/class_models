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

def ConvLayer(x,filters,kernel_size,stride,activation,name_scope):
    with tf.variable_scope(name_scope):
        out = tf.layers.conv3d(x,filters,kernel_size=kernel_size,strides=stride,padding='same',name='conv')
        out = GroupNorm(out,name_scope='gn',group=8)
        if activation is not None:
            out = tf.nn.leaky_relu(out)
    return out

def inception_resnet_block(x,block_type,out_channels,scale,name_scope):
    with tf.variable_scope(name_scope):
        _,_,_,_,C1 = x.get_shape().as_list()
        if block_type=='block35':
            branch0=ConvLayer(x,filters=32,kernel_size=1,stride=1,activation='relu',name_scope='A_b0_l1')
            branch1=ConvLayer(x,filters=32,kernel_size=1,stride=1,activation='relu',name_scope='A_b1_l1')
            branch1=ConvLayer(branch1,filters=32,kernel_size=3,stride=1,activation='relu',name_scope='A_b1_l2')
            branch2=ConvLayer(x,filters=32,kernel_size=1,stride=1,activation='relu',name_scope='A_b2_l1')
            branch2=ConvLayer(branch2,filters=48,kernel_size=3,stride=1,activation='relu',name_scope='A_b2_l2')
            branch2=ConvLayer(branch2,filters=64,kernel_size=3,stride=1,activation='relu',name_scope='A_b2_l3')
            branches=[branch0,branch1,branch2]
        elif block_type=='block17':
            branch0=ConvLayer(x,filters=192,kernel_size=1,stride=1,activation='relu',name_scope='B_b0_l1')
            branch1=ConvLayer(x,filters=128,kernel_size=1,stride=1,activation='relu',name_scope='B_b1_l1')
            branch1=ConvLayer(branch1,filters=160,kernel_size=[1,7,3],stride=1,activation='relu',name_scope='B_b1_l2')
            branch1=ConvLayer(branch1,filters=192,kernel_size=[7,1,3],stride=1,activation='relu',name_scope='B_b1_l3')
            branches=[branch0,branch1]
        elif block_type=='block8':
            branch0=ConvLayer(x,filters=192,kernel_size=1,stride=1,activation='relu',name_scope='C_b0_l1')
            branch1=ConvLayer(x,filters=192,kernel_size=1,stride=1,activation='relu',name_scope='C_b1_l1')
            branch1=ConvLayer(branch1,filters=224,kernel_size=[1,3,3],stride=1,activation='relu',name_scope='C_b1_l2')
            branch1=ConvLayer(branch1,filters=256,kernel_size=[3,1,3],stride=1,activation='relu',name_scope='C_b1_l3')
            branches=[branch0,branch1]
        else:
             raise ValueError('Unknown Inception-ResNet block type.'
                         'Expects "Block35", "Block17" or "Block8",'
                         'but got:' + str(block_type))
        y=tf.concat(branches,axis=-1)
        y=ConvLayer(y,filters=out_channels,kernel_size=3,stride=1,activation='None',name_scope='conv1')
        _,_,_,_,C2 = y.get_shape().as_list()
        if C1!=C2:
            residual=ConvLayer(x,filters=C2,kernel_size=1,stride=1,activation='None',name_scope='conv2')
        else:
            residual=x
        out=tf.nn.leaky_relu(residual+scale*y)
        return out

def InceptionResNetv2(x,num_classes,name_scope):
    with tf.variable_scope(name_scope):
        #stem block   input:[128,128,32,4]
        x=ConvLayer(x,filters=32,kernel_size=3,stride=1,activation='relu',name_scope='conv1')
        x=ConvLayer(x,filters=32,kernel_size=3,stride=2,activation='relu',name_scope='conv2')   #[64,64,16,32]
        #5 Inception-resnet-A
        for i in range(1,6):
            x=inception_resnet_block(x,block_type='block35',out_channels=64,scale=1,name_scope='blockA%s'%(i))   #[64,64,16,64]
        #Reduction-A
        branch0=ConvLayer(x,filters=64,kernel_size=3,stride=2,activation='relu',name_scope='conv3')
        branch1=ConvLayer(x,filters=64,kernel_size=1,stride=1,activation='relu',name_scope='conv4')
        branch1=ConvLayer(branch1,filters=64,kernel_size=3,stride=2,activation='relu',name_scope='conv5')
        branch2=tf.layers.max_pooling3d(x,pool_size=2,strides=2,padding='same')
        x=tf.concat([branch0,branch1,branch2],axis=-1)   #[32,32,8,192]
        #10 Inception-resnet-B
        for i in range(1,11):
            x=inception_resnet_block(x,block_type='block17',out_channels=128,scale=1,name_scope='blockB%s'%(i))   #[32,32,8,128]
        #Reduction-B
        branch0=ConvLayer(x,filters=128,kernel_size=1,stride=1,activation='relu',name_scope='conv6')
        branch0=ConvLayer(branch0,filters=128,kernel_size=3,stride=2,activation='relu',name_scope='conv7')
        branch1=ConvLayer(x,filters=128,kernel_size=1,stride=1,activation='relu',name_scope='conv8')
        branch1=ConvLayer(branch1,filters=128,kernel_size=3,stride=2,activation='relu',name_scope='conv9')
        branch2=tf.layers.max_pooling3d(x,pool_size=2,strides=2,padding='same')
        x=tf.concat([branch0,branch1,branch2],axis=-1)   #[16,16,4,384]
        #5 Inception-resnet-C
        for i in range(1,6):
            x=inception_resnet_block(x,block_type='block8',out_channels=256,scale=1,name_scope='blockC%s'%(i))   #[16,16,4,256]
        #fc
        _,H,W,D,C = x.get_shape().as_list()
        x=tf.layers.average_pooling3d(x,pool_size=(H,W,D),strides=1)  
        x=tf.reshape(x,[-1,C])      #(bs,256)
        #x=tf.layers.dense(x,units=64,activation=tf.nn.sigmoid,name='l1')   #(bs,64)
        out=tf.layers.dense(x,units=num_classes,activation=tf.nn.softmax,name='l2')
        return out
