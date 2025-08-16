import importlib.metadata
__version__ = importlib.metadata.version("tungnaa")

from .model import *
from .text import *
from .util import *
from .split import get_datasets