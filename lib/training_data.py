#!/usr/bin/env python3
""" Process training data for model training """

class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_size, training_opts):
        self.batchsize = 0
        self.model_input_size = model.input_size
        self.model_output_size = model_output_size
        self.training_opts = training_opts
        
        
    def minibatch_ab(self, images, batchsize, side):
        """ Keep a queue filled to 8x Batch Size """
        self.batchsize = batchsize
        training_size = self.training_opts.get("training_size", 256)
        batch_shape = list((
            (batchsize, training_size, training_size, 3),   # sample images
            (batchsize, self.model_input_size, self.model_input_size, 3),
            (batchsize, self.model_output_size, self.model_output_size, 3)))
        
        # TODO: The Original code implemented complicately here, because it concerns
        #       multithreading. Let's simplify here. 
            