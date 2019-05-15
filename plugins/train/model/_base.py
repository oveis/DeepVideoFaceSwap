#!/usr/bin/env python3
""" Base class for Models. """

import logging, os

from lib.model.nn_blocks import NNBlocks

logger = logging.getLogger(__name__)


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self,
                 model_dir,
                 input_shape=None,
                 encoder_dim=None,
                 predict=False):
        
        self.model_dir = model_dir
        self.blocks = NNBlocks()
        self.input_shape = input_shape
        self.encoder_dim = encoder_dim
        self.networks = dict()  # Networks for the model
        self.predict = predict    
    
    
    def build(self):
        self.add_networks()
        self.load_models(swapped=False)
        self.build_autoencoders()
        self.log_summary()
        self.compile_predictors(initialize=True)
        
    
    def add_network(self, network_type, side, network):
        name = network_type.lower() + ('_{}'.format(side.upper()) if side else '')
        filename = 'train_{}.h5'.format(name)
        logger.debug("Add Network. Name: '%s', FileName: '%s'", name, filename)
        self.networks[name] = NNMeta(os.path.join(self.model_dir, filename), network_type, side, network)

        
    def add_predictor(self, side, model):
        self.predictors[side] = model
        if not self.state.inputs:
            self.store_input_shapes(model)
        if not self.output_shape:
            self.set_output_shape(model)
    
    
    def store_input_shapes(self, model):
        pass
    
    
    def set_output_shape(self, model):
        pass
    
        
    def load_models(self, swapped):
        pass

    
    def log_summary(self):
        pass
    
    
    def compile_predictors(self, initialize):
        pass

        
class NNMeta():
    """ Class to hold a neural network and it's meta data """
    
    def __init__(self, filename, network_type, side, network):
        self.filename = filename
        self.type = network_type.lower()
        self.side = side
        self.name = self.type + ('_{}'.format(self.side) if side else '')
        self.network = network
        self.network.name = self.name
        self.config = network.get_config()
        self.weights = network.get_weights()