#!/usr/bin/env python3
""" Process training data for model training """

import logging

from lib.multithreading import FixedProducerDispatcher
from lib.queue_manager import queue_manager

logger = logging.getLogger(__name__)


class TrainingDataGenerator():
    """ Generate training data for models """
    
    def __init__(self, model_input_size, model_output_size, training_opts):
        self.batchsize = 0
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.training_opts = training_opts
        
    
    @staticmethod
    def minibatch(side, load_process):
        """ A generator function that yield epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        for batch_wrapper in load_process:
            with batch_wrapper as batch:
                yield batch
                
        load_process.stop()
        load_process.join()
            
    
    @staticmethod
    def make_queues(side):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_name = 'train_{}'.format(side)
        q_names = ['{}_{}'.format(q_name, direction) for direction in ('in', 'out')]
        queues = [queue_manager.get_queue(queue) for queue in q_names]
        return queues
    
            
    def process_face(self, filename, side):
        """ Load an image and perform transformation and warping """

        image = cv2.imread(filename)
        image = self.processing.color_adjust(image)
        
        image = self.processing.random_transform(image)
        image = self.processing.do_random_flip(image)
        
        sample = image.copy()[:, :, :3]
        
        processed = self.processing.random_warp(image)
        processed.insert(0, sample)
        return processed
        
    
    def load_batches(self, mem_gen, images, side, batchsize):
        """ Load the warped images and target images to queue """
        
        def _img_iter(imgs):
            while True:
                for img in imgs:
                    yield img
                    
        img_iter = _img_iter(images)
        epoch = 0
        for memory_wrapper in mem_gen:
            memory = memory_wrapper.get()
            
            for i, img_path in enumerate(img_iter):
                imgs = self.process_face(img_path, side)
                for j, img in enumerate(imgs):
                    memory[j][i][:] = img
                epoch += 1
                if i == batchsize - 1:
                    break
            memory_wrapper.read()
        
            
    def minibatch_ab(self, images, batchsize, side):
        """ Keep a queue filled to 8x Batch Size """
        print('[TEST] minibatch_ab: batchsize: {}, side: {}'.format(batchsize, side))
        self.batchsize = batchsize
        queue_in, queue_out = self.make_queues(side)
        training_size = self.training_opts.get("training_size", 256)        
        batch_shape = list((
            (batchsize, training_size, training_size, 3),   # sample images
            (batchsize, self.model_input_size, self.model_input_size, 3),
            (batchsize, self.model_output_size, self.model_output_size, 3)))
        
        load_process = FixedProducerDispatcher(
            method=self.load_batches,
            shapes=batch_shape,
            in_queue=queue_in,
            out_queue=queue_out,
            args=(images, side, batchsize))
        
        load_process.start()
        return self.minibatch(side, load_process)
            
            
    
            