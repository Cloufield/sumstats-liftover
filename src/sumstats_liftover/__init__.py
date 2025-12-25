"""
sumstats-liftover: Fast chain-based liftover for pandas DataFrames

A standalone implementation for lifting over genomic coordinates in pandas DataFrames
using UCSC chain files.
"""

from .liftover_df import liftover_df, parse_chain_to_segments, Segments
from .chain_files import get_chain_path, list_chain_files, get_chain_info

__version__ = "1.1.0"
__all__ = [
    'liftover_df',
    'parse_chain_to_segments',
    'Segments',
    'get_chain_path',
    'list_chain_files',
    'get_chain_info',
]

