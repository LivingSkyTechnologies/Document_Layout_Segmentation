import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix


#%% - defining the upsampler
def PSPPooling(inp_mask, filter_size):
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(inp_mask)
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(4,4))(inp_mask)
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(8,8))(inp_mask)
    x4 = tf.keras.layers.MaxPooling2D(pool_size=(16,16))(inp_mask)
    
    x1 = tf.keras.layers.Conv2D(int(filter_size/4), (1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x2 = tf.keras.layers.Conv2D(int(filter_size/4), (1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x3 = tf.keras.layers.Conv2D(int(filter_size/4), (1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x3)
    x3 = tf.keras.layers.Dropout(0.5)(x3)
    x4 = tf.keras.layers.Conv2D(int(filter_size/4), (1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x4)
    x4 = tf.keras.layers.Dropout(0.5)(x4)
    
    x1 = tf.keras.layers.UpSampling2D(size=(2,2))(x1)
    x2 = tf.keras.layers.UpSampling2D(size=(4,4))(x2)
    x3 = tf.keras.layers.UpSampling2D(size=(8,8))(x3)
    x4 = tf.keras.layers.UpSampling2D(size=(16,16))(x4)
    
    x = tf.keras.layers.Concatenate()([x1,x2,x3,x4,inp_mask])
    x = tf.keras.layers.Conv2D(filter_size, (1,1), dilation_rate=(1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    return x

def intermediate_layer(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, dilation_rate=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def build(img_size, output_channels):
    base_model = tf.keras.applications.ResNet50(input_shape=[img_size, img_size, 3], include_top=False)

    layer_names = [
        'conv1_relu',
        'conv2_block2_out',
        'conv2_block3_out',
        'conv3_block3_out',
        'conv3_block4_out',
        'conv4_block5_out',
        'conv4_block6_out',
        'conv5_block3_out'
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    up_stack = [
        pix2pix.upsample(512, 5, apply_dropout=True),
        intermediate_layer(256, 5, apply_dropout=True),
        pix2pix.upsample(256, 5, apply_dropout=True),
        intermediate_layer(128, 5, apply_dropout=True),
        pix2pix.upsample(128, 5, apply_dropout=True),
        intermediate_layer(64, 5, apply_dropout=True),
        pix2pix.upsample(64, 5, apply_dropout=True)
    ]

    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])
  
    bn = tf.keras.layers.BatchNormalization()
    x = bn(inputs)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    i = -1
    for up, skip in zip(up_stack, skips):
        i += 1
        if i % 2 == 1:
            x = up(x)
            continue
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer before PSP
    x = tf.keras.layers.Conv2DTranspose(output_channels, 5, strides=(2,2), padding='same', activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
  
    pooled = tf.keras.layers.Concatenate()([x, inputs])
    pooled = PSPPooling(x, output_channels*2)
    pooled = tf.keras.layers.Dropout(0.5)(pooled)
  
    out = tf.keras.layers.Conv2DTranspose(output_channels, 5, strides=(1,1), padding='same', activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  
    return tf.keras.Model(inputs=inputs, outputs=out)

