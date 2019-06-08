#!/usr/bin/env python3
""" GAN Model """

from keras.layers import Conv2D, Dense, Flatten, Input, Lambda, Reshape, add, concatenate, multiply
from keras.models import Model as KerasModel
from keras.initializers import RandomNormal
from ._base import ModelBase, logger


class Model(ModelBase):
    """ GAN Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.adversarial_autoencoders = dict()  # Adversarial autoencoder models

        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024

        kwargs['trainer'] = 'gan'
        super().__init__(*args, **kwargs)
        
        logger.debug("Initialized %s", self.__class__.__name__)

        
    def build(self):
        super().build()
        self.compile_discriminators()
        self.build_adversarial_autoencoders()
        self.compile_adversarial_autoencoders()


    def add_networks(self):
        """ Add the model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder())
        self.add_network("decoder", "b", self.decoder())
        self.add_network("encoder", None, self.encoder())
        self.add_network("discriminator", "a", self.discriminator())
        self.add_network("discriminator", "b", self.discriminator())
        logger.debug("Added networks")


    def add_adversarial_autoencoder(self, side, model):
        """ Add a adversarial auoencoder """
        logger.debug("Adding adversarial autoencoder: (side: '%s', model: %s)", side, model)
        if self.gpus > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.gpus)
        self.adversarial_autoencoders[side] = model


    def build_adversarial_autoencoders(self):
        """ Initialize adversarial autoencoder model """
        logger.debug("Initializing adversarial autoencoder model")
        img = Input(shape=self.input_shape)
        def one_minus(x): return 1 - x
        
        for side, model in self.predictors.items():
            alpha, reconstructed_img = model(img)
            # masked_img = alpha * reconstructed_img + (1 - alpha) * img
            masked_img = add([multiply([alpha, reconstructed_img]), multiply([Lambda(one_minus)(alpha), img])])
            
            discriminator = self.networks['discriminator_{}'.format(side)].network
            out_discriminator = discriminator(concatenate([masked_img, img], axis=-1))
            
            # The adversarial_autoencoder model (stacked generator and discriminator) takes
            # img as input => generated encoded representation and reconstructed image => determines validity.
            adversarial_autoencoder = KerasModel(img, [reconstructed_img, out_discriminator])
            self.add_adversarial_autoencoder(side, adversarial_autoencoder)
            
        logger.debug("Initialized adversarial autoencoder model")

        
    def build_autoencoders(self):
        """ Initialize autoencoder models """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
#         if self.config.get("mask_type", None):
#         mask_shape = (self.input_shape[:2] + (1, ))
#         inputs.append(Input(shape=mask_shape, name="mask"))

        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")


    def compile_predictors(self, initialize=True):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.predictors.items():
            model.compile(loss=['mae', 'mse'], optimizer=optimizer)

            loss_names = ['Loss_D_{}'.format(side), 'Loss_G_{}'.format(side)]

            if initialize:
                self.state.add_session_loss_names(side, loss_names)
                self.history['Loss_D_{}'.format(side)] = list()
                self.history['Loss_G_{}'.format(side)] = list()

        logger.debug("Compiled Predictors")


    def compile_discriminators(self):
        """ Compile the discriminators """
        logger.debug("Compiling Discriminators")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side in ("a", "b"):
            model = self.networks["discriminator_{}".format(side)].network
            model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        logger.debug("Compiled Discriminators.")


    def compile_adversarial_autoencoders(self):
        """ Compile the adversarial autoencoders """
        logger.debug("Compiling Adversarial Autoencoders")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.adversarial_autoencoders.items():
            model.compile(loss=['mae', 'mse'],
                          loss_weights=[1, 0.5],
                          optimizer=optimizer)

        logger.debug("Compiled Adversarial Autoencoders")

        
    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(1024)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        out = self.blocks.upscale_ps(var_x, 512, RandomNormal(0, 0.02))
        return KerasModel(input_, out)

    
    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale_ps(var_x, 256, RandomNormal(0, 0.02))
        var_x = self.blocks.upscale_ps(var_x, 128, RandomNormal(0, 0.02))
        var_x = self.blocks.upscale_ps(var_x, 64, RandomNormal(0, 0.02))
        var_x = self.blocks.res_block_gan(var_x, 64)
        var_x = self.blocks.res_block_gan(var_x, 64)

        alpha = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        rgb = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(var_x)

        return KerasModel(input_, outputs=[alpha, rgb])

    
    def discriminator(self):
        """ Discriminator Network """
        input_ = Input(shape=(self.input_shape[0],
                              self.input_shape[1],
                              self.input_shape[2]*2))

        var_x = self.blocks.conv_d_gan(input_, 64)
        var_x = self.blocks.conv_d_gan(var_x, 128)
        var_x = self.blocks.conv_d_gan(var_x, 256)
        out = Conv2D(1, kernel_size=4, kernel_initializer=RandomNormal(0, 0.02),
                       use_bias=False, padding="same", activation="sigmoid")(var_x)

        model = KerasModel(inputs=[input_], outputs=out)
        # For the adversarial autoencoder, we won't train discriminator
        model.trainable = False
        return model