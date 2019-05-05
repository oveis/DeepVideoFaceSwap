#!/usr/bin python3
import keras.layers import Dense, Flatten, Input, Reshape
import keras.model import Model as KerasModel

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D


IMAGE_SHAPE = (64, 64, 3)

class Model:
    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyRelu(0.1)(x)
            return x
        return block
    
    
    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(X)
            return x
        return block
    
                
    def encoder(self):
        x_input = Input(IMAGE_SHAPE)
        x = x_input
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(1024)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale(512)(x)
        return KerasModel(inputs=x_input, outputs=x)
    
    
    def decoder(self):
        x_input = Input((8, 8, 512))
        x = x_input
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activateion='sigmoid')(x)
        return KerasModel(inputs=x_input, outputs=x)
        
        
    
    