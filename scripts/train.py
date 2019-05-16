#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging, os

from plugins.train.model.original import Model as OriginalModel
from plugins.train.trainer.original import Trainer as OriginalTrainer

logger = logging.getLogger(__name__)

# Model directory. This is where the training data will be stored.
MODEL_DIR = '../output/model'


class Train():
    def __init__(self, arguments):
        logger.debug("Initializing Train Model")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        self.args = arguments
        self.images = self.get_images()

    def process(self):
        logger.debug("Starting Training Process")
        self.training()
        
        
    def training(self):
        try:
            model = self.load_model()
            trainer = self.load_trainer(model)
            self.run_training_cycle(mode, trainer)
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
            
    
    def load_model(self):
        logger.debug("Loading Model")
        model = OriginalModel(
            model_dir,
            predict=False)
        return model
    
    
    def load_trainer(self, model):
        logger.debug("Loading Trainer")
        trainer = OriginalTrainer(mode,
                                  self.images,
                                  self.args.batch_size)
        return trainer
        
    
    def run_training_cycle(self, model, trainer):
        for iteration in range(0, self.args.iterations):
            logger.trace('Training iteration: %s', iteration)
            trainer.train_one_step()
            
        model.save_models()
        trainer.clear_tensorboard()
        

    def get_images(self):
        images = dict()
        for side = ("a", "b"):
            image_dir = getattr(self.args, "input_{}".format(side))
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                exit(1)

            if not os.listdir(image_dir):
                logger.error("Error: '%s' contains no images", image_dir)
                exit(1)

            images[side] = get_image_paths(image_dir)
        logger.info("Model A Directory: %s", self.args.input_a)
        logger.info("Model B Directory: %s", self.args.input_b)
        return images
        
        
    
    