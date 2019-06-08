#!/usr/bin/env python3
""" GAN Trainer """

import logging
import numpy as np

from ._base import TrainerBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Trainer(TrainerBase):
    """ GAN Trainer """
    
    def __init__(self, model, images, batch_size):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        
        super().__init__(model, images, batch_size)
        
        self.use_mixup = True
        self.mixup_alpha = 0.2
        
        logger.debug("Initialized %s", self.__class__.__name__)


    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        do_preview = False if viewer is None else True
        do_timelapse = False if timelapse_kwargs is None else True

        # ---------------------
        #  Train Discriminators
        # ---------------------
        
        # Select a random half batch of images
        batch_A = self.batchers['a'].get_next(do_preview)
        warped_A = batch_A[0]
        target_A = batch_A[1]

        batch_B = self.batchers['b'].get_next(do_preview)
        warped_B = batch_B[0]
        target_B = batch_B[1]
        
        # Generate a half batch of new images
        gen_alphasA, gen_imgsA = self.model.predictors['a'].predict(warped_A)
        gen_alphasB, gen_imgsB = self.model.predictors['b'].predict(warped_B)
        
        # gen_masked_imgsA = gen_alphasA * gen_imgsA + (1 - gen_alphasA) * warped_A
        # gen_masked_imgsB = gen_alphasB * gen_imgsB + (1 - gen_alphasB) * warped_B
        gen_masked_imgsA = np.array([gen_alphasA[i] * gen_imgsA[i] + (1 - gen_alphasA[i]) * warped_A[i]
                                     for i in range(self.batch_size)])
        gen_masked_imgsB = np.array([gen_alphasB[i] * gen_imgsB[i] + (1 - gen_alphasB[i]) * warped_B[i]
                                     for i in range(self.batch_size)])

        discriminatorA = self.model.networks["discriminator_a"].network
        discriminatorB = self.model.networks["discriminator_a"].network

        valid = np.ones((self.batch_size, ) + discriminatorA.output_shape[1:])
        fake = np.zeros((self.batch_size, ) + discriminatorB.output_shape[1:])

        concat_real_inputA = np.array([np.concatenate([target_A[i], warped_A[i]], axis=-1)
                                      for i in range(self.batch_size)])
        concat_real_inputB = np.array([np.concatenate([target_B[i], warped_B[i]], axis=-1)
                                      for i in range(self.batch_size)])
        concat_fake_inputA = np.array([np.concatenate([gen_masked_imgsA[i], warped_A[i]], axis=-1)
                                      for i in range(self.batch_size)])
        concat_fake_inputB = np.array([np.concatenate([gen_masked_imgsB[i], warped_B[i]], axis=-1)
                                      for i in range(self.batch_size)])
        
        if self.use_mixup:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixup_A = lam * concat_real_inputA + (1 - lam) * concat_fake_inputA
            mixup_B = lam * concat_real_inputB + (1 - lam) * concat_fake_inputB
            
        # Train the discriminators
        if self.use_mixup:
            d_lossA = discriminatorA.train_on_batch(mixup_A, lam * valid)
            d_lossB = discriminatorB.train_on_batch(mixup_B, lam * valid)
        else:
            d_lossA = discriminatorA.train_on_batch(
                np.concatenate([concat_real_inputA, concat_fake_inputA], axis=0),
                np.concatenate([valid, fake], axis=0))

            d_lossB = discriminatorB.train_on_batch(
                np.concatenate([concat_real_inputB, concat_fake_inputB], axis=0),
                np.concatenate([valid, fake], axis=0))
            
        # ---------------------
        #  Train Discriminators
        # ---------------------
        
        # Train the generators
        g_lossA = self.model.adversarial_autoencoders['a'].train_on_batch(warped_A, [target_A, valid])
        g_lossB = self.model.adversarial_autoencoders['b'].train_on_batch(warped_B, [target_B, valid])
        
        if do_preview:
            self.samples.images['a'] = self.batchers['a'].compile_sample(self.batch_size)
            self.samples.images['b'] = self.batchers['b'].compile_sample(self.batch_size)
            
        if do_timelapse:
            self.timelapse.get_sample('a', timelapse_kwargs)
            self.timelapse.get_sample('b', timelapse_kwargs)
            
        self.model.state.increment_iterations()

        loss = {'Loss_D_a': [d_lossA[0]],
                'Loss_D_b': [d_lossB[0]],
                'Loss_G_a': [g_lossA[0]],
                'Loss_G_b': [g_lossB[0]]}

        for key, val in loss.items():
            self.store_history(key, val)

        self.log_tensorboard('a', [d_lossA[0], g_lossA[0]])
        self.log_tensorboard('b', [d_lossB[0], g_lossB[0]])

        self.print_loss(loss)
            
        if do_preview:
            samples = self.samples.show_sample()
            if samples is not None:
                viewer(samples, "Training - 'S': Save Now. 'ENTER': Save and Quit")

        if do_timelapse:
            self.timelapse.output_timelapse()


    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        logger.trace(loss)

        output = list()
        for key, val in loss.items():
            output.append('{}: {:.5f}'.format(key, val[0]))
        output = ', '.join(output)

        print("[{}] [#{:05d}] {}".format(self.timestamp, self.model.iterations, output), end='\r')
