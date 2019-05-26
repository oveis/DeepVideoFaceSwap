#!/usr/bin/env python3

import argparse
import logging

from scripts.data_collector import FaceImageCollector
from scripts.train import Train

logger = logging.getLogger(__name__)


class DeepVideoFaceSwap:
    """ Deep Video Face Swap """
    
    def preprocess(self, celebrity, output_dir, limit):
        collector = FaceImageCollector()
        collector.collect(celebrity, output_dir, limit)
        

    def train(self, trainer_name, batch_size, iterations, input_a, input_b, model_dir, num_gpu):
        try:
            train = Train(trainer_name, batch_size, iterations, input_a, input_b, model_dir, num_gpu)
            train.process()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            pass
        except Exception:
            logger.exception('Got Exception on train process')
    
    
    def convert(self):
        raise NotImplementedError()


def set_log_level(log_level):
    if (log_level == 'warning'):
        level = logging.WARNING
    elif (log_level == 'info'):
        level = logging.INFO
    elif (log_level == 'debug'):
        level = logging.DEBUG
    else:
        level = logging.ERROR

    logging.basicConfig(filename='log_{}.log'.format(log_level), level=level)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['preprocess', 'train', 'convert'])
    parser.add_argument('--log', choices=['error', 'warning', 'info', 'debug'], default='error')
    
    # Preprocess arguments
    parser.add_argument('--celebrity', default=None)
    parser.add_argument('--output-dir', default='dataset')
    parser.add_argument('--limit', type=int, default=100)
    
    # Train arguments
    parser.add_argument('--trainer-name', default='original')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--input-A', help="Person A's face image dataset to train")
    parser.add_argument('--input-B', help="Person B's face image dataset to train")
    parser.add_argument('--model-dir', help="Model directory where the training data will be stored")
    parser.add_argument('--num-gpu', type=int, default=1)
    
    # Convert arguments
    
    args = parser.parse_args()
    
    face_swap = DeepVideoFaceSwap()
    set_log_level(args.log)
    
    if args.task == 'preprocess':
        face_swap.preprocess(args.celebrity, args.output_dir, args.limit)
    elif args.task == 'train':
        face_swap.train(args.trainer_name, args.batch_size, args.iterations, args.input_A, args.input_B,
                       args.model_dir, args.num_gpu)
    elif args.task == 'convert':
        face_swap.convert()