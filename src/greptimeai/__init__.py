import logging

from .collection import Collector

# TODO(yuanbohan): is it needed to pass variables in Collector construction?
# only one instance, DO NOT initiate other Collector
collector = Collector()

logger = logging.getLogger("greptimeai")
