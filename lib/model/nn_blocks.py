#!/usr/bin/env python3
""" Neural Network Blocks """

import logging

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from .layers import PixelShuffler

logger = logging.getLogger(__name__)


class NNBlocks():
    def __init__(self):
        logger.debug("Initialized %s", self.__class__.__name__)
    
    
    def conv(self, inp, filters, kernel_size=5, strides=2, padding='same'):
        var_x = Conv2D(filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding)(inp)
        return var_x
    
    
    def upscale(self, inp, filters, kernel_size=3, padding='same'):
        var_x = Conv2D(filters * 4,
                     kernel_size=kernel_size,
                     padding=padding)(inp)
        var_x = LeakyReLU(0.1)(var_x)
        var_x = PixelShuffler()(var_x)
        return var_x