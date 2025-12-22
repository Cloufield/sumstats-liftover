"""
sumstats-liftover: Fast chain-based liftover for pandas DataFrames

A standalone implementation for lifting over genomic coordinates in pandas DataFrames
using UCSC chain files.
"""

from .liftover_df import liftover_df, parse_chain_to_segments, Segments

__version__ = "0.1.0"
__all__ = ['liftover_df', 'parse_chain_to_segments', 'Segments']

