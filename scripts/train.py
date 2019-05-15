#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging, os

from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)

# Model directory. This is where the training data will be stored.
MODEL_DIR = '../output/model'


class Train():
    def __init__(self):
        logger.debug("Initializing Train Model")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

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
        model = PluginLoader.get_model(self.trainer_name)(
            model_dir,
            predict=False)
        return model
    
    
    def load_trainer(self, model):
        logger.debug("Loading Trainer")
        trainer = PluginLoader.get_trainer(model.trainer)
        trainer = trainer(mode, 
                          self.images, 
                          self.args.batch_size)
        return trainer
        
        

        
        
    
    