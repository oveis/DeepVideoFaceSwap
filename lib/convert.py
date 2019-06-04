#!/usr/bin/env python3
""" Converter for faceswap.py """

import logging
import numpy as np

from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)


class Converter():
    """ Swap a source face with a target """
    def __init__(self, output_dir, output_size, mask_type):
        self.output_dir = output_dir
        self.scale = 1
        self.adjustments = dict(box=None, mask=None)
        self.load_plugins(output_size, mask_type)
        logger.debug("Initialized %s", self.__class__.__name__)
        

    def load_plugins(self, output_size, mask_type):
        """ Load the requested adjustment plugins """
        self.adjustments['box'] = PluginLoader.get_converter('mask', 'box_blend')(
            'none',
            output_size)
        
        self.adjustments['mask'] = PluginLoader.get_converter('mask', 'mask_blend')(
            mask_type,
            output_size,
            False)
        
        logger.debug('Loaded plugins: {}'.format(self.adjustments))


    def process(self, in_queue, out_queue):
        """ Process items from the queue """
        while True:
            item = in_queue.get()
            if item == 'EOF':
                in_queue.put(item)
                break
                
            try:
                image = self.patch_image(item)
            except Exceptiono:
                logger.exception('Failed to convert image: {}'.format(item['filename']))
                image = item['image']
                
            out_queue.put((item['filename'], image))


    def patch_image(self, predicted):
        """ Patch the image """
        frame_size = (predicted['image'].shape[1], predicted['image'].shape[0])
        new_image = self.get_new_image(predicted, frame_size)
        patched_face = self.post_warp_adjustments(predicted, new_image)
        patched_face = self.scale_image(patched_face)
        patched_face = np.rint(patched_face * 255.0).astype('uint8')
        return patched_face
    
    
    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        
        
    def post_warp_adjustments(self, predicted, new_image):
        """ Apply fixes to the image after warping """
        
        
    def scale_image(self, frame):
        """ Scale the image if requested """
        
                                 