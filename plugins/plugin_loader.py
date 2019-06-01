#!/usr/bin/env python3

import logging
from importlib import import_module

logger = logging.getLogger(__name__)


class PluginLoader():
    
    @staticmethod
    def get_model(name):
        """ Return requested model plugin """
        return PluginLoader._import("train.model", name)
    
    
    @staticmethod
    def get_trainer(name):
        """ Return requested trainer plugin """
        return PluginLoader._import("train.trainer", name)
    
    
    @staticmethod
    def get_converter(category, name):
        """ Return the converter sub plugin """
        return PluginLoader._import('convert.{}'.format(category), name)
    
    
    @staticmethod
    def _import(attr, name):
        """ Import the plugin's module """
        name = name.replace('-', '_')
        title = attr.split('.')[-1].title()
        mod = '.'.join(('plugins', attr, name))
        module = import_module(mod)
        return getattr(module, title)