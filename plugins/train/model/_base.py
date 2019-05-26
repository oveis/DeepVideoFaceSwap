#!/usr/bin/env python3
""" Base class for Models. """

import logging
import os
import time

from json import JSONDecodeError

from keras import losses
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import get_custom_objects, multi_gpu_model

from lib import Serializer
from lib.model.losses import DSSIMObjective
from lib.model.nn_blocks import NNBlocks
# from lib.multithreading import MultiThread

DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.999

logger = logging.getLogger(__name__)


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self,
                 model_dir,
                 num_gpu,
                 input_shape=None,
                 encoder_dim=None,
                 training_image_size=256,
                 trainer_name='original',
                 predict=False):
        
        self.model_dir = model_dir
        self.num_gpu = num_gpu
        self.blocks = NNBlocks()
        self.input_shape = input_shape
        self.output_shape = None          # set after model is compiled
        self.encoder_dim = encoder_dim
        self.trainer_name = trainer_name
        self.predict = predict
        
        self.state = State(self.model_dir,
                           self.trainer_name,
                           training_image_size)
    
        self.networks = dict()            # Networks for the model
        self.predictors = dict()          # Predictors for model
        self.history = dict()             # Loss history per save iteration
        
        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = {'training_size': training_image_size}

        self.build()
            
    
#     @property
#     def config_section(self):
#         """ The section name for loading config """
#         return '.'.join(self.__module__.split('.')[-2:])
    
    
#     @property
#     def config(self):
#         """ Return config dict for current plugin """
#         global _CONFIG
#         if not _CONFIG:
#             model_name = self.config_section
#             _CONFIG = Config(model_name).config_dict
#         return _CONFIG
    
    
    @property
    def iterations(self):
        """ Get current training iteration number """
        return self.state.iterations
    
    
    @property
    def models_exist(self):
        """ Return if all files exist and clear session """
        return all([os.path.isfile(model.filename) for model in self.networks.values()])
    
    
    def add_network(self, network_type, side, network):
        """ Add a NNMeta object """
        filename = '{}_{}'.format(self.trainer_name, network_type.lower())
        name = network_type.lower()
        
        if side:
            side = side.lower()
            filename += '_{}'.format(side.upper())
            name += '_{}'.format(side)
            
        filename += '.h5'
        logger.debug("Add Network. Name: '%s', FileName: '%s'", name, filename)
        
        name = network_type.lower() + ('_{}'.format(side.lower()) if side else '')
        filename = 'train_{}.h5'.format(name)
        self.networks[name] = NNMeta(os.path.join(self.model_dir, filename), network_type, side, network)

    
    def store_input_shapes(self, model):
        """ Store the input and output shapes to state for model """
        self.state.inputs = {tensor.name: K.int_shape(tensor)[-3:] for tensor in model.inputs}
    
    
    def set_output_shape(self, model):
        out = [K.int_shape(tensor)[-3:] for tensor in model.outputs]
        if not out:
            raise ValueError("No output found! Check your model.")
        self.output_shape = tuple(out[0])

        
    def add_predictor(self, side, model):
        """ Add a predictor to the predictors dictionary """
        if self.num_gpu > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.num_gpu)
            
        self.predictors[side] = model
        if not self.state.inputs:
            self.store_input_shapes(model)
        if not self.output_shape:
            self.set_output_shape(model)
        
        
    def map_models(self, swapped):
        """ Map the models for A/B side for swapping """
        models_map = {'a': dict(), 'b': dict()}
        sides = ('a', 'b') if not swapped else ('b', 'a')
        
        for network in self.networks.values():
            if network.side == sides[0]:
                models_map['a'][network.type] = network.filename
            else:
                models_map['b'][network.type] = network.filename
       
        return models_map
        
        
    def load_models(self, swapped):
        """ Load models from file """
        if not self.models_exist:
            if self.predict:
                logger.error('No model found in folder [{}]'.format(self.model_dir))
                raise FileNotFoundError()
            else:
                return None

        model_mapping = self.map_models(swapped)
        for network in self.networks.values():
            if not network.side:
                is_loaded = network.load()
            else:
                is_loaded = network.load(fullpath=model_mapping[network.side][network.type])
            if not is_loaded:
                break
                
    
    def save_models(self):
        """ Save models """
#         for network in self.networks.values():
#             name = 'save_{}'.format(network.name)
#             save_threads.append(MultiThread(network.save, name=name))
        
#         save_threads.append(MultiThread(self.state.save, name='save_state'))
        
#         for thread in save_threads:
#             thread.start()
            
#         for thread in save_threads:
#             if thread.has_error:
#                 logger.error(thread.errors[0])
#             thread.join()

        for network in self.networks.values():
            network.save()
        
        self.state.save()
        
    
    def loss_function(self, side, initialize):
        """ Set the loss function """
        # TODO: Try use various loss functions.
        loss_func = DSSIMObjective()
#         loss_func = losses.mean_absolute_error

        return loss_func
        
        
    def compile_predictors(self, initialize=True):
        """ Compile the predictors """
#         learning_rate = self.config.get("learning_rate", DEFAULT_LEARNING_RATE)
        optimizer = Adam(lr=DEFAULT_LEARNING_RATE, beta_1=DEFAULT_BETA_1, beta_2=DEFAULT_BETA_2)
        
        for side, model in self.predictors.items():
            loss_names = ['loss']
            loss_funcs = [self.loss_function(side, initialize)]
        
            model.compile(optimizer=optimizer, loss=loss_funcs)
            
            if initialize:
                self.state.add_session_loss_names(side, loss_names)
                self.history[side] = list()
    
    
    def build(self):
        self.add_networks()
        self.load_models(swapped=False)
        self.build_autoencoders()
        self.compile_predictors(initialize=True)

        
class NNMeta():
    """ Class to hold a neural network and it's meta data """
    
    def __init__(self, filename, network_type, side, network):
        self.filename = filename
        self.type = network_type.lower()
        self.side = side
        self.name = self.type + ('_{}'.format(self.side) if side else '')
        self.network = network
        self.network.name = self.name
        self.config = network.get_config()    # For pingpong restore
        self.weights = network.get_weights()  # For pingpong restore
        
    
    def load(self, fullpath=None):
        """ Load model """
        fullpath = fullpath if fullpath else self.filename
        try:
            # Jinil note:
            # The original code(https://github.com/deepfakes/faceswap/blob/master/plugins/train/model/_base.py#L557)
            # uses 'self.filename' instead of 'fullpath', but I think it's a bug. 
            network = load_model(fullpath, custom_objects=get_custom_objects())
        except Exception:
            logger.exception('Failed to load existing training data')
            return False
        self.config = network.get_config()
        self.network = network               # Update network with saved model
        self.network.name = self.type
        return True
        
        
    def save(self, fullpath=None):
        """ Save model """
        fullpath = fullpath if fullpath else self.filename
        self.weights = self.network.get_weights()
        self.network.save(fullpath)
        
    
class State():
    """ Class to hold the model's current state and autoencoder structure """
    def __init__(self, model_dir, model_name, training_image_size):
        self.serializer = Serializer.get_serializer("json")
        filename = "{}_state.{}".format(model_name, self.serializer.ext)
        self.filename = str(model_dir / filename)
        self.name = model_name
        self.iterations = 0
        self.session_iterations = 0
        self.training_size = training_image_size
        self.sessions = dict()
        self.lowest_avg_loss = dict()
        self.inputs = dict()
        self.config = dict()
        self.load()
        self.session_id = self.new_session_id()
        self.create_new_session()
        

    @property
    def face_shapes(self):
        """ Return a list of stored face shape inputs """
        return [tutple(val) for key, val in self.inputs.items() if key.startswith('face')]
    
    
    @property
    def mask_shapes(self):
        """ Return a list of stored mask shape inputs """
        return [tuple(val) for key, val in self.inputs.items() if key.startswith('mask')]
    
    
    @property
    def loss_name(self):
        """ Return the loss names for this session """
        return self.sessions[self.session_id]['loss_names']
    
    
    @property
    def current_session(self):
        """ Return the current session dict """
        return self.sessions[self.session_id]
    
    
    def new_session_id(self):
        """ Return new session_id """
        if not self.sessions:
            return 1
        else:
            return max(int(key) for key in self.sessions.keys()) + 1

        
    def create_new_session(self):
        """ Create a new session """
        self.sessions[self.session_id] = {'timestamp': time.time(),
                                          'loss_names': dict(),
                                          'batchsize': 0,
                                          'iterations': 0}
        
        
    def add_session_loss_names(self, side, loss_names):
        """ Add the session loss names to the sessions dictionary """
        self.sessions[self.session_id]['loss_names'][side] = loss_names
        
        
    def add_session_batchsize(self, batchsize):
        """ Add the session batchsize to the sessions dictionary """
        self.sessions[self.session_id]['batchsize'] = batchsize
        
        
    def increment_iterations(self):
        """ Increment total and session iterations """
        self.iterations += 1
        self.sessions[self.session_id]['iterations'] += 1

        
    def load(self):
        """ Load state file """
        try:
            with open(self.filename, "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode('utf-8'))
                self.name = state.get('name', self.name)
                self.sessions = state.get('sessions', dict())
                self.lowest_avg_loss = state.get('lowest_avg_loss', dict())
                self.iterations = state.get('iterations', 0)
                self.training_size = state.get('training_size', 256)
                self.inputs = state.get('inputs', dict())
                self.config = state.get('config', dict())
        except IOError as err:
            logger.warning('No existing state file found.')
        except JSONDecodeError:
            logger.exception('JSONDecodeError')
            
    
    def save(self):
        """ Save iteration number to state file """
        try:
            with open(self.filename, 'wb') as out:
                state = {'name': self.name,
                         'sessions': self.sessions,
                         'lowest_avg_loss': self.lowest_avg_loss,
                         'iterations': self.iterations,
                         'inputs': self.inputs,
                         'training_size': self.training_size}
                
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode('utf-8'))
        except IOError as err:
            logger.exception('Unable to save model state')
                         
                
        
        
    
        
