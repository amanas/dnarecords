"""
DNARecords package init.
"""

from importlib_metadata import version
from . import reader, writer, helper, macros

__version__ = version("dnarecords")
__all__ = [reader, writer, helper, macros]
