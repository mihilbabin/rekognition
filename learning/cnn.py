from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.models import Sequential

import tensorflow as tf

class CNN:
    def __init__(self, classes_num=7, input_shape=(48, 48, 1)):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes_num, activation='softmax'))

class MultilayerCNN:
    def __init__(self, classes_num=7, input_shape=(48, 48, 1)):
        img_input = tf.keras.layers.Input(shape=input_shape)

        conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv_1')(img_input)
        maxpool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_1)
        x = tf.keras.layers.BatchNormalization()(maxpool_1)
        
        conv_2a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', name='conv_2a')(x)
        conv_2b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', name='conv_2b')(conv_2a)
        maxpool_2a = tf.keras.layers.MaxPooling2D((3,3), strides=(1, 1), name='maxpool_2a')(x)
        conv_2c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_2c')(maxpool_2a)
        concat_1 = tf.keras.layers.concatenate(inputs=[conv_2b, conv_2c], axis=3, name='concat2')
        maxpool_2b = tf.keras.layers.MaxPooling2D((3,3), strides=(2, 2), name='maxpool_2b')(concat_1)

        conv_3a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', name='conv_3a')(maxpool_2b)
        conv_3b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', name='conv_3b')(conv_3a)
        maxpool_3a = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), name='maxpool_3a')(maxpool_2b)
        conv_3c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_3c')(maxpool_3a)
        concat_3 = tf.keras.layers.concatenate(inputs=[conv_3b, conv_3c],axis=3,name='concat3')
        maxpool_3b = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), name='maxpool_3b')(concat_3)
     
        net = tf.keras.layers.Flatten()(maxpool_3b)
        net = tf.keras.layers.Dense(classes_num, activation='softmax', name='predictions')(net)
        
        self.model = tf.keras.Model(img_input, net, name='fer')
