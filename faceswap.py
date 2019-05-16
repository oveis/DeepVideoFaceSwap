#!/usr/bin/env python3

from scripts.train import Train


def execute_train(batch_size, iterations, input_a, input_b):
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
        
    
if __name__ == '__main__":
    execute_train(10, 10, 'dataset/input_a', 'dataset/input_b')