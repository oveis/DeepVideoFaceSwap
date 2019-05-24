#!/usr/bin/python
""" Logging Setup """

import logging
from logging.handlers import QueueHandler
from lib.queue_manager import queue_manager

LOG_QUEUE = queue_manager._log_queue


def set_root_logger(loglevel=logging.INFO, queue=LOG_QUEUE):
    """ Setup the root logger.
        Loaded in main process and into any spawned processes
        Automatically added in multithreading.py"""
    rootlogger = logging.getLogger()
    q_handler = QueueHandler(queue)
    rootlogger.addHandler(q_handler)
    rootlogger.setLevel(loglevel)