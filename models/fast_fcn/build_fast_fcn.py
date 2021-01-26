import tensorflow as tf


def JPU(conv3, conv4, conv5, width=512):
    conv5 = tf.keras.layers.Conv2D(2048, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.ReLU()(conv5)
    
    conv4 = tf.keras.layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.ReLU()(conv4)
    
    conv3 = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    
    feats = [conv5, conv4, conv3]
    feats[-2] = tf.keras.layers.UpSampling2D(size=(2,2))(feats[-2])
    feats[-3] = tf.keras.layers.UpSampling2D(size=(4,4))(feats[-3])
    
    dialation1 = tf.keras.Sequential()
    dialation1.add(tf.keras.layers.SeparableConv2D(width, 3, dilation_rate=(1,1), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    dialation1.add(tf.keras.layers.BatchNormalization())
    dialation1.add(tf.keras.layers.ReLU())
    
    dialation2 = tf.keras.Sequential()
    dialation2.add(tf.keras.layers.SeparableConv2D(width, 3, dilation_rate=(2,2), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    dialation2.add(tf.keras.layers.BatchNormalization())
    dialation2.add(tf.keras.layers.ReLU())
    
    dialation3 = tf.keras.Sequential()
    dialation3.add(tf.keras.layers.SeparableConv2D(width, 3, dilation_rate=(4,4), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    dialation3.add(tf.keras.layers.BatchNormalization())
    dialation3.add(tf.keras.layers.ReLU())
    
    dialation4 = tf.keras.Sequential()
    dialation4.add(tf.keras.layers.SeparableConv2D(width, 3, dilation_rate=(8,8), use_bias=False, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    dialation4.add(tf.keras.layers.BatchNormalization())
    dialation4.add(tf.keras.layers.ReLU())
    
    feat = tf.keras.layers.Concatenate()([feats[0], feats[1], feats[2]])
    feat = tf.keras.layers.Concatenate()([dialation1(feat), dialation2(feat), dialation3(feat), dialation4(feat)])
    
    return feat

def build(img_size, output_channels):
    base_model = tf.keras.applications.ResNet50(input_shape=[img_size, img_size, 3], include_top=False)

    layer_names = [
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out'
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])
    
    base_outputs = down_stack(inputs)
   
    c5 = base_outputs[-1]
    c4 = base_outputs[-2]
    c3 = base_outputs[-3]
    
    feat = JPU(c3, c4, c5)
    
    out = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(feat)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Conv2D(output_channels, 1, padding='same', activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(out)
    out = tf.keras.layers.UpSampling2D(size=(8, 8))(out)
    
    return  tf.keras.Model(inputs=inputs, outputs=out)

