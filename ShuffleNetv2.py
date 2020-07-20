import tensorflow as tf

def ChannelShuffle(x,group):
    _,H,W,D,C = x.get_shape().as_list()
    channel_per_group=C//group
    x=tf.reshape(x,[-1,H,W,D,group,channel_per_group])
    x=tf.transpose(x,[0,1,2,3,5,4])
    _,H,W,D,C,G=x.get_shape().as_list()
    x=tf.reshape(x,[-1,H,W,D,C*G])
    return x

def GroupNorm(x,name_scope,group=4,esp=1e-5):
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
        out = GroupNorm(out,name_scope='gn',group=4)
        if activation is not None:
            out = tf.nn.leaky_relu(out)
    return out

def ShuffleBlock(x,unit_type,name_scope):
    with tf.variable_scope(name_scope):
        _,H,W,D,C = x.get_shape().as_list()
        if unit_type==1:
            branch1=ConvLayer(x,filters=C,kernel_size=3,stride=2,activation='relu',name_scope='type1_conv1')
            branch2=ConvLayer(x,filters=C//2,kernel_size=1,stride=1,activation='relu',name_scope='type1_conv2')
            branch2=ConvLayer(branch2,filters=C//2,kernel_size=3,stride=2,activation='relu',name_scope='type1_conv3')
            branch2=ConvLayer(branch2,filters=C,kernel_size=1,stride=1,activation='relu',name_scope='type1_conv4')
            x=tf.concat([branch1,branch2],axis=-1)
            x=ChannelShuffle(x,group=2)
        else:
            x1=x[:,:,:,:,:C//2]
            x2=x[:,:,:,:,C//2:]
            x2=ConvLayer(x2,filters=C//2,kernel_size=1,stride=1,activation='relu',name_scope='type2_conv1')
            x2=ConvLayer(x2,filters=C//2,kernel_size=3,stride=1,activation='relu',name_scope='type2_conv2')
            x=tf.concat([x1,x2],axis=-1)
            x=ChannelShuffle(x,group=2)
    return x

def ShuffleNet(x,num_classes,name_scope):
    with tf.variable_scope(name_scope):
        out_channels=[24,48,96,192]
        # input[128,128,32,4]
        x=ConvLayer(x,filters=out_channels[0],kernel_size=3,stride=2,activation='relu',name_scope='conv1') #[64,64,16,24]
        for i in range(1,5):
            if i==1:
                x=ShuffleBlock(x,unit_type=1,name_scope='blockA%s'%str(i))
            else:
                x=ShuffleBlock(x,unit_type=2,name_scope='blockA%s'%str(i))  #[32,32,8,48]
        for i in range(1,9):
            if i==1:
                x=ShuffleBlock(x,unit_type=1,name_scope='blockB%s'%str(i))
            else:
                x=ShuffleBlock(x,unit_type=2,name_scope='blockB%s'%str(i))  #[16,16,4,96]
        for i in range(1,5):
            if i==1:
                x=ShuffleBlock(x,unit_type=1,name_scope='blockC%s'%str(i))
            else:
                x=ShuffleBlock(x,unit_type=2,name_scope='blockC%s'%str(i))  #[8,8,2,192]
        _,H,W,D,C = x.get_shape().as_list()
        x=tf.layers.average_pooling3d(x,pool_size=(H,W,D),strides=1)  
        x=tf.reshape(x,[-1,C])      #(bs,192)
        out=tf.layers.dense(x,units=num_classes,activation=tf.nn.softmax,name='fc')
    return out
        