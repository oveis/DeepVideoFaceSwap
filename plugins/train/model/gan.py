#!/usr/bin/env python3
""" GAN Model """

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, add, multiply

from keras.models import Model as KerasModel

from ._base import ModelBase, logger


class Model(ModelBase):
    """ GAN Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        if "input_shape" not in kwargs:
            kwargs["input_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024

        kwargs['trainer'] = 'gan'
        kwargs["is_gan"] = True
        super().__init__(*args, **kwargs)
        
        logger.debug("Initialized %s", self.__class__.__name__)

        
    def add_networks(self):
        """ Add the model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder())
        self.add_network("decoder", "b", self.decoder())
        self.add_network("encoder", None, self.encoder())
        self.add_network("discriminator", "a", self.discriminator())
        self.add_network("discriminator", "b", self.discriminator())
        logger.debug("Added networks")

    
    def build_adversarial_autoencoders(self):
        """ Initialize adversarial autoencoder model """
        logger.debug("Initializing adversarial autoencoder model")
        img = Input(shape=self.img_shape)
        def one_minus(x): return 1 - x
        
        for side, model in self.predictors.items():
            alpha, reconstructed_img = model(img)
            # masked_img = alpha * reconstructed_img + (1 - alpha) * img
            masked_img = add([multiply([alpha, reconstructed_img]), multiply([Lambda(one_minus)(alpha), img])])
            
            discriminator = self.networks['discriminator_{}'.format(side)]
            out_discriminator = discriminator(concatenate([masked_img], axis=-1))
            
            # The adversarial_autoencoder model (stacked generator and discriminator) takes
            # img as input => generated encoded representation and reconstructed image => determines validity.
            adversarial_autoencoder = KerasModel(img, [reconstructed_img, out_discriminator])
            self.add_adversarial_autoencoder(side, adversarial_autoencoder)
            
        logger.debug("Initialized adversarial autoencoder model")

        
    def build_autoencoders(self):
        """ Initialize autoencoder models """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = (self.input_shape[:2] + (1, ))
            inputs.append(Input(shape=mask_shape, name="mask"))

        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

        
    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        if not self.config.get("lowmem", False):
            var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512)
        return KerasModel(input_, var_x)

    
    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, 128)
            var_y = self.blocks.upscale(var_y, 64)
            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)

    
    def discriminator(self):
        """ Discriminator Network """
        input_ = Input(shape=(shape=(self.input_shape[0], 
                                     self.input_shape[1], 
                                     self.input_shape[2]*2)))
        var_x = self.blocks.conv_d_gan(var_x, 64)
        var_x = self.blocks.conv_d_gan(var_x, 128)
        var_x = self.blocks.conv_d_gan(var_x, 256)
        var_x = Conv2D(1, kernel_size=4, kernel_initializer="he_normal",
                       use_bias=False, padding="same", activation="sigmoid")(var_x)
        model = KerasModel(inputs=[input_], outputs=var_x)
        # For the adversarial autoencoder, we won't train discriminator
        model.trainable = False
        return model