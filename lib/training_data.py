#!/usr/bin/env python3
""" Process training data for model training """

import logging
import cv2
import numpy as np

from lib.multithreading import FixedProducerDispatcher
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama


logger = logging.getLogger(__name__)


class TrainingDataGenerator():
    """ Generate training data for models """
    
    def __init__(self, model_input_size, model_output_size, training_opts):
        self.batchsize = 0
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size
        self.training_opts = training_opts
        self.processing = ImageManipulation(model_input_size,
                                            model_output_size,
                                            training_opts.get('coverage_ratio', 0.625))
        logger.debug("Initialized %s", self.__class__.__name__)
        
        
    @staticmethod
    def minibatch(side, load_process):
        """ A generator function that yield epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Launching minibatch generator for queue (side: '%s')", side)
        for batch_wrapper in load_process:
            with batch_wrapper as batch:
                yield batch
                
        load_process.stop()
        load_process.join()
            

    @staticmethod
    def make_queues(side):
        """ Create the buffer token queues for Fixed Producer Dispatcher """
        q_names = ['train_{}_{}'.format(side, direction) 
                   for direction in ('in', 'out')]
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
        
    
    def load_batches(self, mem_gen, images, side, batchsize=0):
        """ Load the warped images and target images to queue """
        logger.debug("Loading batch: (image_count: %s, side: '%s')", len(images), side)
        
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
            memory_wrapper.ready()
        
            
    def minibatch_ab(self, images, batchsize, side):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s'", len(images), batchsize, side)
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


class ImageManipulation():
    """ Manipulations to be performed on training images """

    def __init__(self, input_size, output_size, coverage_ratio):
        """ input_size: Size of the face input into the model
            output_size: Size of the face that comes out of the modell
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (input_size: %s, output_size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, input_size, output_size, coverage_ratio)
        # Transform args
        self.rotation_range = 10  # Range to randomly rotate the image by
        self.zoom_range = 0.05  # Range to randomly zoom the image by
        self.shift_range = 0.05  # Range to randomly translate the image by
        self.random_flip = 0.5  # Chance to flip the image horizontally
        # Transform and Warp args
        self.input_size = input_size
        self.output_size = output_size
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale
        logger.debug("Initialized %s", self.__class__.__name__)


    @staticmethod
    def color_adjust(img):
        """ Color adjust RGB image """
        logger.debug("Color adjusting image")
        return img.astype('float32') / 255.0


    @staticmethod
    def separate_mask(image):
        """ Return the image and the mask from a 4 channel image """
        mask = None
        if image.shape[2] == 4:
            logger.debug("Image contains mask")
            mask = np.expand_dims(image[:, :, -1], axis=2)
            image = image[:, :, :3]
        else:
            logger.debug("Image has no mask")
        return image, mask


    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.debug("Coverage: %s", coverage)
        return coverage


    def random_transform(self, image):
        """ Randomly transform an image """
        logger.debug("Randomly transforming image")
        height, width = image.shape[0:2]

        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        tnx = np.random.uniform(-self.shift_range, self.shift_range) * width
        tny = np.random.uniform(-self.shift_range, self.shift_range) * height

        mat = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            (width // 2, height // 2), rotation, scale)
        mat[:, 2] += (tnx, tny)
        result = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (width, height),
            borderMode=cv2.BORDER_REPLICATE)  # pylint: disable=no-member

        logger.debug("Randomly transformed image")
        return result


    def do_random_flip(self, image):
        """ Perform flip on image if random number is within threshold """
        logger.debug("Randomly flipping image")
        if np.random.random() < self.random_flip:
            logger.debug("Flip within threshold. Flipping")
            retval = image[:, ::-1]
        else:
            logger.debug("Flip outside threshold. Not Flipping")
            retval = image
        logger.debug("Randomly flipped image")
        return retval


    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
        logger.debug("Randomly warping image")
        height, width = image.shape[0:2]
        coverage = self.get_coverage(image)
        assert height == width and height % 2 == 0

        range_ = np.linspace(height // 2 - coverage // 2,
                             height // 2 + coverage // 2,
                             5, dtype='float32')
        mapx = np.broadcast_to(range_, (5, 5)).copy()
        mapy = mapx.T
        # mapx, mapy = np.float32(np.meshgrid(range_,range_)) # instead of broadcast

        pad = int(1.25 * self.input_size)
        slices = slice(pad // 10, -pad // 10)
        dst_slice = slice(0, (self.output_size + 1), (self.output_size // 4))
        interp = np.empty((2, self.input_size, self.input_size), dtype='float32')
        ####

        for i, map_ in enumerate([mapx, mapy]):
            map_ = map_ + np.random.normal(size=(5, 5), scale=self.scale)
            interp[i] = cv2.resize(map_, (pad, pad))[slices, slices]  # pylint: disable=no-member

        warped_image = cv2.remap(  # pylint: disable=no-member
            image, interp[0], interp[1], cv2.INTER_LINEAR)  # pylint: disable=no-member
        logger.debug("Warped image shape: %s", warped_image.shape)

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_slice, dst_slice]
        mat = umeyama(src_points, True, dst_points.T.reshape(-1, 2))[0:2]
        target_image = cv2.warpAffine(  # pylint: disable=no-member
            image, mat, (self.output_size, self.output_size))
        logger.debug("Target image shape: %s", target_image.shape)

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        if target_mask is None:
            logger.debug("Randomly warped image")
            return [warped_image, target_image]

        logger.debug("Target mask shape: %s", target_mask.shape)
        logger.debug("Randomly warped image and mask")
        return [warped_image, target_image, target_mask]
    

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.debug("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image)

        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')

        face_core = cv2.convexHull(np.concatenate(  # pylint: disable=no-member
            [source[17:], destination[17:]], axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core,  # pylint: disable=no-member
                                          (pty, ptx),
                                          False) >= 0:
                    indicies_to_remove.add(idx)

        for idx in sorted(indicies_to_remove, reverse=True):
            source.pop(idx)
            destination.pop(idx)

        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear")
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped_image = cv2.remap(image,  # pylint: disable=no-member
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,  # pylint: disable=no-member
                                 cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        target_image = image

        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage // 2, size // 2 + coverage // 2)
#        slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(  # pylint: disable=no-member
            warped_image[slices, slices, :], (self.input_size, self.input_size),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.debug("Warped image shape: %s", warped_image.shape)
        target_image = cv2.resize(  # pylint: disable=no-member
            target_image[slices, slices, :], (self.output_size, self.output_size),
            cv2.INTER_AREA)  # pylint: disable=no-member
        logger.debug("Target image shape: %s", target_image.shape)

        warped_image, warped_mask = self.separate_mask(warped_image)
        target_image, target_mask = self.separate_mask(target_image)

        if target_mask is None:
            logger.debug("Randomly warped image")
            return [warped_image, target_image]

        logger.debug("Target mask shape: %s", target_mask.shape)
        logger.debug("Randomly warped image and mask")
        return [warped_image, target_image, target_mask]