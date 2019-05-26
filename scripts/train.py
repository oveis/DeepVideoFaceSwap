#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging
import os
import cv2

from plugins.plugin_loader import PluginLoader
from lib.utils import get_image_paths, get_folder

logger = logging.getLogger(__name__)


class Train():
    def __init__(self, trainer_name, batch_size, iterations, input_a, input_b, model_dir, num_gpu=1):
        logger.debug("Initializing Train Model")
        self.trainer_name = trainer_name
        self.batch_size = batch_size
        self.iterations = iterations
        self.input_a = input_a
        self.input_b = input_b
        self.model_dir = model_dir
        self.num_gpu = num_gpu
        
        self.images = self._get_images()

        
    @property
    def image_size(self):
        """ Get the training set image size for sorting in model data """
        image = cv2.imread(self.images['a'][0])
        size = image.shape[0]
        logger.debug("Training image size: %s", size)
        return size
    
    
    def process(self):
        logger.debug("Starting Training Process")
        self._training()
        
        
    def _get_images(self):
        images = dict()
        for side in ("a", "b"):
            image_dir = getattr(self, "input_{}".format(side))
            if not os.path.isdir(image_dir):
                err_msg = "Error: '{}' does not exist".format(image_dir)
                logger.error(err_msg)
                raise NotADirectoryError(err_msg)

            if not os.listdir(image_dir):
                err_msg = "Error: '{}' contains no images".format(image_dir)
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            images[side] = get_image_paths(image_dir)
            
        logger.info("Model A Directory: %s", self.input_a)
        logger.info("Model B Directory: %s", self.input_b)
        return images
    
    
    def _load_model(self):
        logger.debug("Loading Model: {}".format(self.trainer_name))
        model_dir = get_folder(self.model_dir)
        model = PluginLoader.get_model(self.trainer_name)
        model = model(model_dir, 
                      self.num_gpu,
                      training_image_size=self.image_size,
                      predict=False)
        return model
    
    
    def _load_trainer(self, model):
        logger.debug("Loading Trainer")
        trainer = PluginLoader.get_trainer(model.trainer_name)
        trainer = trainer(model,
                          self.images,
                          self.batch_size)
        return trainer
        
    
    def _run_training_cycle(self, model, trainer):
        for iteration in range(self.iterations):
            logger.info('Training iteration: %s', iteration)
            trainer.train_one_step()
            
        model.save_models()
        

    def _training(self):
        try:
            model = self._load_model()
            trainer = self._load_trainer(model)
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save_models()
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            exit(0)
        except Exception as err:
            raise err
            
            
    
        
        
    
    