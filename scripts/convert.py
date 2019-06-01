#!/usr/bin python3
""" The script to run the convert process of faceswap """

import logging
import multiprocessing as mp

from lib.convert import Converter
from lib.utils import get_folder
from multiprocessing import Pool
from scripts.fsmedia import Images
from tqdm import tqdm


logger = logging.getLogger(__name__)

CONVERT_IN_QUEUE = 'convert_in'
CONVERT_OUT_QUEUE = 'convert_out'
PATCH_QUEUE = 'patch'


class Convert():
    """ The convert process """
    def __init__(self, input_video, output_dir):
        logger.debug("Initialized %s", self.__class__.__name__)
        self.add_queues()
        
        self.images = Images(input_video)
        self.disk_io = DiskIO(self.images,
                              queue_manager.get_queue(CONVERT_IN_QUEUE),
                              queue_manager.get_queue(CONVERT_OUT_QUEUE),
                              output_dir, 
                              input_video)
        self.predictor = Predict()            # TODO
        self.converter = Converter(get_folder(output_dir),
                                   self.predictor.output_size)
    
    
    @property
    def queue_size(self):
        """ Set queue size to double number of cpu available """
        return mp.cpu_count() * 2
    
    
    def add_queues(self):
        for q_name in (CONVERT_IN_QUEUE, CONVERT_OUT_QUEUE, PATCH_QUEUE):
            queue_manager.add_queue(q_name, self.queue_size)
        

    def get_processes(self):
        """ Get the number of processes to use """
        running_processes = len(mp.active_children())
        return max(mp.cpu_count() - running_processes, 1)
        
        
    def convert_images(self):
        """ Convert the images """
        save_queue = queue_manager.get_queue(CONVERT_OUT_QUEUE)
        patch_queue = queue_manager.get_queue(PATCH_QUEUE)
        save_queue.put('EOF')

        pool = Pool(processes=self.get_processes())
        pool.apply_async(self.converter.process, args=[patch_queue, save_queue])
        
        pool.close()
        pool.join()

        
    def process(self):
        """ Process the conversion """
        logger.debug('Starting Conversion')
        
        self.convert_images()
        self.disk_io.save_thread.join()
        queue_manager.terminate_queues()
        logger.debug('Completed Conversion')
        

class DiskIO():
    """ Load images from disk and get the detected faces
        Save images back to disk """
    def __init__(self, images, convert_in_queue, convert_out_queue, output_dir, input_video):
        self.images = images
        self.load_queue = convert_in_queue
        self.save_queue = convert_out_queue
        self.output_dir = output_dir
        self.input_video = input_video
        self.writer = self.get_writer()
        logger.debug("Initialized %s", self.__class__.__name__)
    
    
    @property
    def total_count(self):
        """ Return the total number of frames to be converted """
        return self.images.images_found


    def get_writer(self):
        """ Return the writer plugin """
        return PluginLoader.get_converter('writer', 'ffmpeg')(
            self.output_dir, self.total_count, self.input_video)
        
        
    def get_detected_faces(self, filename, image):
        # TODO
        raise NotImplementedError()
        
        
    def load(self):
        """ Load the images with detected faces"""
        logger.debug('Load Images with detected faces')
        for filename, image in self.images.load():
            detected_faces = self.get_detected_faces(filename, image)
            item = dict(filename=filename, image=image, detected_faces=detected_faces)
            self.load_queue.put(item)
            
        self.load_queue.put('EOF')
        
        
    def save(self):
        """ Save the converted images """
        logger.debug('Save Images: Start')
        for _ in tqdm(range(self.total_count), desc='Converting', file=sys.stdout):
            item = self.save_queue.get()
            if item == 'EOF':
                break
            filename, image = item
            self.writer.write(filename, image)
        
        self.writer.close()
        logger.debug('Save Faces: Complete')
                     
        
        
        
class Predict():
    """ Predict faces from incoming queue """
    def __init__(self, in_queue, queue_size):
        self.batchsize = min(queue_size, 16)
        self.in_queue = in_queue
        self.out_queue = queue_manager
        logger.deubug("Inpitialized %s: (out_queue: %s)", self.__class__.__name__, self.out_queue)
        
        
    
        