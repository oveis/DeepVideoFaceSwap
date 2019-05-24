#!/usr/bin/env python3
"""
Library providing convenient classes and methods for writing data to files.
"""
import logging
import json

logger = logging.getLogger(__name__)


class Serializer():
    """ Parent Serializer class """
    ext = ""
    woptions = ""
    roptions = ""

    
    @classmethod
    def marshal(cls, input_data):
        """ Override for marshalling """
        raise NotImplementedError()

        
    @classmethod
    def unmarshal(cls, input_string):
        """ Override for unmarshalling """
        raise NotImplementedError()
        

class JSONSerializer(Serializer):
    """ JSON Serializer """
    ext = "json"
    woptions = "w"
    roptions = "r"

    
    @classmethod
    def marshal(cls, input_data):
        return json.dumps(input_data, indent=2)

    
    @classmethod
    def unmarshal(cls, input_string):
        return json.loads(input_string)



def get_serializer(serializer):
    """ Return requested serializer """
    if serializer == 'json':
        return JSONSerializer
    else:
        raise NotImplementedError()