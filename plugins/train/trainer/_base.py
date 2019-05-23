#!/usr/bin/env python3
""" Base Trainer Class """

import logging
from lib.training_data import TrainingDataGenerator

logger = logging.getLogger(__name__)

class TrainerBase():
    def __init__(self, model, images, batch_size):
        self.batch_size = batch_size
        self.model = model
        self.model.state.add_session_batchsize(batch_size)
        self.images = images
        self.sides = sorted(key for key in self.images.keys())
        
        self.batchers = {side: Batcher(side,
                                       images[side],
                                       self.model,
                                       batch_size)
                         for side in self.sides}
    
    
    def train_one_step(self):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        loss = dict()
        for side, batcher in self.batchers.items():
            loss[side] = batcher.train_one_batch()
        
        self.model.state.increment_iterations()
        
        for side, side_loss in loss.items():
            self.store_history(side, side_loss)
            
            
    def store_history(self, side, loss):
        logger.trace("Updating loss history: '%s'", side)
        self.model.history[side].append(loss[0])
        

class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, batch_size):
        self.model = model
        self.side = side
        self.feed = self.load_generator().minibatch_ab(images, batch_size, self.side)
        
    
    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        generator = TrainingDataGenerator(input_size, output_size, self.model.training_opts)
        return generator
    
    
    def train_one_batch(self):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        batch = self.get_next()
        loss = self.model.predictors[self.side].train_on_batch(*batch)
        loss = loss if isinstance(loss, list) else [loss]
        return loss
    
    
    def get_next(self):
        """ Return the next batch from the generator
            Items should come out as: (warped, target [, mask]) """
        batch = next(self.feed)
        batch = batch[1:]    # Remove full size samples from batch. 
                             # TODO: Use sample image when we need.
        return batch