#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import logging
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float

import queue as Queue
import sys
import threading
import numpy as np
from lib.logger import LOG_QUEUE, set_root_logger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_launched_processes = set() # pylint: disable=invalid-name



class ConsumerBuffer():
    """ Memory buffer for consuming """
    def __init__(self, dispatcher, index, data):
        logger.debug("Initializing %s: (dispatcher: '%s', index: %s, data: %s)",
                     self.__class__.__name__, dispatcher, index, data)
        self._data = data
        self._id = index
        self._dispatcher = dispatcher
        logger.debug("Initialized %s", self.__class__.__name__)

    def get(self):
        """ Return Data """
        return self._data

    def free(self):
        """ Return Free """
        self._dispatcher.free(self._id)

    def __enter__(self):
        """ On Enter """
        return self.get()

    def __exit__(self, *args):
        """ On Exit """
        self.free()


class WorkerBuffer():
    """ Memory buffer for working """
    def __init__(self, index, data, stop_event, queue):
        logger.debug("Initializing %s: (index: '%s', data: %s, stop_event: %s, queue: %s)",
                     self.__class__.__name__, index, data, stop_event, queue)
        self._id = index
        self._data = data
        self._stop_event = stop_event
        self._queue = queue
        logger.debug("Initialized %s", self.__class__.__name__)

    def get(self):
        """ Return Data """
        return self._data

    def ready(self):
        """ Worker Ready """
        if self._stop_event.is_set():
            return
        self._queue.put(self._id)

    def __enter__(self):
        """ On Enter """
        return self.get()

    def __exit__(self, *args):
        """ On Exit """
        self.ready()


class FixedProducerDispatcher():
    """
    Runs the given method in N subprocesses
    and provides fixed size shared memory to the method.
    This class is designed for endless running worker processes
    filling the provided memory with data,
    like preparing trainingsdata for neural network training.
    As soon as one worker finishes all worker are shutdown.
    Example:
        # Producer side
        def do_work(memory_gen):
            for memory_wrap in memory_gen:
                # alternative memory_wrap.get and memory_wrap.ready can be used
                with memory_wrap as memory:
                    input, exp_result = prepare_batch(...)
                    memory[0][:] = input
                    memory[1][:] = exp_result
        # Consumer side
        batch_size = 64
        dispatcher = FixedProducerDispatcher(do_work, shapes=[
            (batch_size, 256,256,3), (batch_size, 256,256,3)])
        for batch_wrapper in dispatcher:
            # alternative batch_wrapper.get and batch_wrapper.free can be used
            with batch_wrapper as batch:
                send_batch_to_trainer(batch)
    """
    CTX = mp.get_context("spawn")
    EVENT = CTX.Event

    def __init__(self, method, shapes, in_queue, out_queue,
                 args=tuple(), kwargs={}, ctype=c_float, workers=1, buffers=None):
        logger.debug("Initializing %s: (method: '%s', shapes: %s, args: %s, kwargs: %s, "
                     "ctype: %s, workers: %s, buffers: %s)", self.__class__.__name__, method,
                     shapes, args, kwargs, ctype, workers, buffers)
        if buffers is None:
            buffers = workers * 2
        else:
            assert buffers >= 2 and buffers > workers
        self.name = "%s_FixedProducerDispatcher" % str(method)
        self._target_func = method
        self._shapes = shapes
        self._stop_event = self.EVENT()
        self._buffer_tokens = in_queue
        for i in range(buffers):
            self._buffer_tokens.put(i)
        self._result_tokens = out_queue
        worker_data, self.data = self._create_data(shapes, ctype, buffers)
        proc_args = {
            'data': worker_data,
            'stop_event': self._stop_event,
            'target': self._target_func,
            'buffer_tokens': self._buffer_tokens,
            'result_tokens': self._result_tokens,
            'dtype': np.dtype(ctype),
            'shapes': shapes,
            'log_queue': LOG_QUEUE,
            'log_level': logger.getEffectiveLevel(),
            'args': args,
            'kwargs': kwargs
        }
        self._worker = tuple(self._create_worker(proc_args) for _ in range(workers))
        self._open_worker = len(self._worker)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _np_from_shared(shared, shapes, dtype):
        """ Numpy array from shared memory """
        arrs = []
        offset = 0
        np_data = np.frombuffer(shared, dtype=dtype)
        for shape in shapes:
            count = np.prod(shape)
            arrs.append(np_data[offset:offset+count].reshape(shape))
            offset += count
        return arrs

    def _create_data(self, shapes, ctype, buffers):
        """ Create data """
        buffer_size = int(sum(np.prod(x) for x in shapes))
        dtype = np.dtype(ctype)
        data = tuple(RawArray(ctype, buffer_size) for _ in range(buffers))
        np_data = tuple(self._np_from_shared(arr, shapes, dtype) for arr in data)
        return data, np_data

    def _create_worker(self, kwargs):
        """ Create Worker """
        logger.debug('Create worker: kwargs: [{}]'.format(kwargs))
        return self.CTX.Process(target=self._runner, kwargs=kwargs)

    def free(self, index):
        """ Free memory """
        if self._stop_event.is_set():
            return
        if isinstance(index, ConsumerBuffer):
            index = index.index
        self._buffer_tokens.put(index)

    def __iter__(self):
        """ Iterator """
        return self

    def __next__(self):
        """ Next item """
        return self.next()

    def next(self, block=True, timeout=None):
        """
        Yields ConsumerBuffer filled by the worker.
        Will raise StopIteration if no more elements are available OR any worker is finished.
        Will raise queue.Empty when block is False and no element is available.
        The returned data is safe until ConsumerBuffer.free() is called or the
        with context is left. If you plan to hold on to it after that make a copy.
        This method is thread safe.
        """
        logger.debug('FixedProducerDispatcher next (block: {}, timeout: {})'.format(block, timeout))
        if self._stop_event.is_set():
            raise StopIteration
        i = self._result_tokens.get(block=block, timeout=timeout)
        if i is None:
            self._open_worker -= 1
            raise StopIteration
        if self._stop_event.is_set():
            raise StopIteration
        return ConsumerBuffer(self, i, self.data[i])

    def start(self):
        """ Start Workers """
        for process in self._worker:
            logger.debug('[TEST] self._worker process: {}'.format(process))
            process.start()
        _launched_processes.add(self)

    def is_alive(self):
        """ Check workers are alive """
        for worker in self._worker:
            if worker.is_alive():
                return True
        return False

    def join(self):
        """ Join Workers """
        self.stop()
        while self._open_worker:
            if self._result_tokens.get() is None:
                self._open_worker -= 1
        while True:
            try:
                self._buffer_tokens.get(block=False, timeout=0.01)
            except Queue.Empty:
                break
        for worker in self._worker:
            worker.join()

    def stop(self):
        """ Stop Workers """
        self._stop_event.set()
        for _ in range(self._open_worker):
            self._buffer_tokens.put(None)

    def is_shutdown(self):
        """ Check if stop event is set """
        return self._stop_event.is_set()

    @classmethod
    def _runner(cls, data=None, stop_event=None, target=None,
                buffer_tokens=None, result_tokens=None, dtype=None,
                shapes=None, log_queue=None, log_level=None,
                args=None, kwargs=None):
        """ Shared Memory Object runner """
        # Fork inherits the queue handler, so skip registration with "fork"
        set_root_logger(log_level, queue=log_queue)
        logger.debug("FixedProducerDispatcher worker for %s started", str(target))
        np_data = [cls._np_from_shared(d, shapes, dtype) for d in data]

        def get_free_slot():
            while not stop_event.is_set():
                i = buffer_tokens.get()
                if stop_event.is_set() or i is None or i == "EOF":
                    break
                yield WorkerBuffer(i, np_data[i], stop_event, result_tokens)

        args = tuple((get_free_slot(),)) + tuple(args)
        try:
            target(*args, **kwargs)
        except Exception as ex:
            print('Running worker is failed: {}'.format(ex))
            logger.exception(ex)
            stop_event.set()
        result_tokens.put(None)
        logger.debug("FixedProducerDispatcher worker for %s shutdown", str(target))
