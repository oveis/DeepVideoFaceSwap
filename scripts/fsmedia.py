#!/usr/bin/env python3

import logging
import cv2
import os


logger = logging.getLogger(__name__)


class Images():
    """ Holds the full frames/images """
    def __init__(self, input_video):
        self.input_video = input_video
        logger.debug("Initialized %s", self.__class__.__name__)


    @property
    def images_found(self):
        """ Number of frames """
        cap = cv2.VideoCapture(self.input_video)
        retval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return retval
    
    
    def load(self):
        """ Load an image and yield it with it's filename """
        iterator = self.load_video_frames
        for filename, image in iterator():
            yield filename, image
            
            
    def load_video_frames(slef):
        """ Return frames from a video file """
        logger.debug('Capturing frames')
        video_name = os.path.splitext(os.path.basename(self.input_video))[0]
        cap = cv2.VideoCapture(self.input_video)
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug('Video terminated')
                break
            idx += 1
            filename = '{}_{:06d}.png'.format(video_name, idx)
            logger.debug('Loading video frame: {}'.format(filename))
            yield filename, frame
            
        cap.release()