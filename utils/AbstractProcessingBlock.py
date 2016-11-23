"""
Enacpsulates all the methods required by a processing block in the system.

This is intended as a base class and should never be directly instantiated.

"""
import logging

class AbstractProcessingBlock(object):
    """ Abstract block class for recognition system"""
    block_name = ""

    def __init__(self, config, blockName=__name__):
        """ Creates a new instance of this block layer """
        self.config = config
        self.block_name = blockName

    def run(self, inputs):
        """ Executes this block and returns the result """
        raise NotImplementedError()

    def log_info(self, message):
        """ Logs a message to the event log """
        logging.info("%s: %s", self.block_name, message)

    def log_warning(self, message):
        """ Logs a warning to the event log """
        logging.warn("%s: %s", self.block_name, message)

    def log_error(self, message):
        """ Logs an error to the event log """
        logging.error("%s: %s", self.block_name, message)
