#!/usr/bin/env python3

from scripts.data_collector import FaceImageCollector
from scripts.train import Train


class DeepVideoFaceSwap:
    """ Deep Video Face Swap """
    
    def preprocess(self, celebrity, output_dir, limit):
        collector = FaceImageCollector()
        collector.collect(celebrity, output_dir, limit)
        

    def train(self, batch_size, iterations, input_a, input_b):
        arguments = {
            'batch_size': batch_size,
            'iterations': iterations,
            'input_a': input_a,
            'input_b': input_b
        }
        try:
            train = Train(arguments)
            train.process()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            pass
        except Exception:
            logger.exception('Got Exception on train process')
    
    
    def convert(self):
        pass
    
    
if __name__ == '__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['preprocess', 'train', 'convert'])
    
    # Preprocess arguments
    parser.add_argument('--celebrity', default=None)
    parser.add_argument('--output-dir', default='dataset')
    parser.add_argument('--limit', type=int, default=100)
    
    # Train arguments
    parser.add_argument('--input-A', help="Person A's face image dataset to train")
    parser.add_argument('--input-B', help="Person B's face image dataset to train")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=10)
    
    # Convert arguments
    
    args = parser.parse_args()
    
    face_swap = DeepVideoFaceSwap()
    
    if args.task == 'preprocess':
        face_swap.preprocess(args.celebrity, args.output_dir, args.limit)
    elif args.task == 'train':
        face_swap.train(args.batch_size, args.iterations, args.input_A, args.input_B)
    elif args.task == 'convert':
        face_swap.convert()