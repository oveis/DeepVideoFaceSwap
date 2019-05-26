#!/usr/bin python3

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel
from ._base import ModelBase, logger


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing Train Model")
        
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (64, 64, 3)
        if 'encoder_dim' not in kwargs:
            kwargs['encoder_dim'] = 1024
        
        kwargs['trainer_name'] = 'original'
        super().__init__(*args, **kwargs)
        
        
    def add_networks(self):
        self.add_network('decoder', 'a', self.decoder())
        self.add_network('decoder', 'b', self.decoder())
        self.add_network('encoder', None, self.encoder())
        
    
    def build_autoencoders(self):
        inputs = [Input(shape=self.input_shape, name='face')]
        
        for side in ('a', 'b'):
            decoder = self.networks['decoder_{}'.format(side)].network
            output = decoder(self.networks['encoder'].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
            
                
    def encoder(self):
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512)
        return KerasModel(inputs=input_, outputs=var_x)
    
    
    def decoder(self):
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        outputs = [var_x]
        return KerasModel(inputs=input_, outputs=outputs)
        
        
    
    