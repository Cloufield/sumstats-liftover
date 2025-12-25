"""
liftover_df: Fast chain-based liftover for pandas DataFrames

A standalone implementation for lifting over genomic coordinates in pandas DataFrames
using UCSC chain files. This is a fast, vectorized implementation that directly parses
UCSC chain files and performs coordinate conversion.

Example usage:
    >>> import pandas as pd
    >>> from sumstats_liftover import liftover_df
    >>> 
    >>> df = pd.DataFrame({
    ...     'CHR': [1, 1, 2],
    ...     'POS': [725932, 725933, 100000],
    ...     'EA': ['G', 'A', 'C'],
    ...     'NEA': ['A', 'G', 'T']
    ... })
    >>> 
    >>> result = liftover_df(
    ...     df,
    ...     chain_path="/path/to/hg19ToHg38.over.chain.gz",
    ...     chrom_col="CHR",
    ...     pos_col="POS"
    ... )
"""

from __future__ import annotations

import gzip
import heapq
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Chain parsing + fast index
# ----------------------------

@dataclass(frozen=True)
class Segments:
    """
    Data structure for storing chain file segments and best disjoint cover.
    
    This class stores both the original chain segments (which may overlap) and
    a disjoint interval cover that selects the highest-scoring segment at each
    position. The disjoint cover enables fast O(log n) coordinate lookup via
    binary search, as each position maps to exactly one segment.
    """
    # Original ungapped segments (can overlap across chains)
    t0: np.ndarray        # int64, start (0-based)
    t1: np.ndarray        # int64, end (0-based, half-open)
    q0: np.ndarray        # int64, query start on the chain's qStrand coordinate system
    score: np.ndarray     # int64
    qsize: np.ndarray     # int64
    qrev: np.ndarray      # bool  (True if qStrand == '-')
    qname: np.ndarray     # object (string per segment)

    # Best disjoint cover (by score) over t0/t1 coordinate
    # These intervals are non-overlapping and cover the coordinate space.
    # When multiple segments overlap, the highest-scoring one is selected.
    bt0: np.ndarray       # int64, start positions of disjoint intervals (0-based)
    bt1: np.ndarray       # int64, end positions of disjoint intervals (0-based, half-open)
    bseg: np.ndarray      # int32, indices into (t0,t1,q0,...) arrays for each interval


def _open_text(path: str):
    """Open a text file, handling gzip compression."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _is_standard_chromosome(chrom: str) -> bool:
    """
    Check if a chromosome name represents a standard chromosome.
    
    Standard chromosomes are: 1-22, X, Y, M/MT (and their 'chr' prefixed versions).
    Filters out alternate contigs, unplaced sequences, etc.
    
    Parameters
    ----------
    chrom : str
        Chromosome name to check
        
    Returns
    -------
    bool
        True if it's a standard chromosome, False otherwise
    """
    chrom_str = str(chrom).strip()
    # Remove 'chr' prefix (case-insensitive)
    if chrom_str.lower().startswith('chr'):
        chrom_str = chrom_str[3:]
    
    # Check for standard chromosomes
    # Standard autosomes: 1-22
    # Standard sex chromosomes: X, Y
    # Standard mitochondrial: M, MT
    # Also accept numeric special chromosomes: 23, 24, 25
    if chrom_str in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                     "21", "22", "23", "24", "25",
                     "X", "Y", "M", "MT", "x", "y", "m", "mt"]:
        return True
    
    # Check if it's a numeric chromosome 1-22
    try:
        chrom_num = int(chrom_str)
        if 1 <= chrom_num <= 22:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def _normalize_chrom_name(chrom: str) -> str:
    """
    Normalize chromosome name by removing 'chr' prefix and handling special cases.
    Converts numeric special chromosomes to string format for matching.
    
    Examples:
        'chr1' -> '1'
        '1' -> '1'
        'chrX' -> 'X'
        'X' -> 'X'
        '23' -> 'X' (for matching with chain files)
        'chrM' -> 'M'
        'M' -> 'M'
        '25' -> 'M' (for matching with chain files)
    """
    chrom_str = str(chrom).strip()
    # Remove 'chr' prefix (case-insensitive)
    if chrom_str.lower().startswith('chr'):
        chrom_str = chrom_str[3:]
    # Convert numeric special chromosomes to string format for matching
    # (chain files use "X", "Y", "M", not "23", "24", "25")
    if chrom_str == "23":
        chrom_str = "X"
    elif chrom_str == "24":
        chrom_str = "Y"
    elif chrom_str == "25":
        chrom_str = "M"
    return chrom_str


def _normalize_chrom_name_vectorized(chroms: np.ndarray) -> np.ndarray:
    """
    Vectorized version of _normalize_chrom_name for numpy arrays.
    
    Parameters
    ----------
    chroms : np.ndarray
        Array of chromosome names (can be object dtype with strings)
    
    Returns
    -------
    np.ndarray
        Array of normalized chromosome names
    """
    # Convert to string array if needed
    if chroms.dtype != object:
        chroms = chroms.astype(str)
    
    # Use pandas string operations for vectorized processing
    chrom_series = pd.Series(chroms, dtype=str)
    
    # Remove 'chr' prefix (case-insensitive) using vectorized string operations
    chrom_series = chrom_series.str.replace(r'^[Cc][Hh][Rr]', '', regex=True)
    
    # Strip whitespace
    chrom_series = chrom_series.str.strip()
    
    # Convert numeric special chromosomes: 23->X, 24->Y, 25->M
    chrom_series = chrom_series.replace({"23": "X", "24": "Y", "25": "M"})
    
    return chrom_series.to_numpy(dtype=object)


def _is_standard_chromosome_vectorized(chroms: np.ndarray) -> np.ndarray:
    """
    Vectorized version of _is_standard_chromosome for numpy arrays.
    
    Parameters
    ----------
    chroms : np.ndarray
        Array of chromosome names (can be object dtype with strings)
    
    Returns
    -------
    np.ndarray
        Boolean array indicating which chromosomes are standard
    """
    # Convert to string array if needed
    if chroms.dtype != object:
        chroms = chroms.astype(str)
    
    chrom_series = pd.Series(chroms, dtype=str)
    
    # Remove 'chr' prefix (case-insensitive)
    chrom_series = chrom_series.str.replace(r'^[Cc][Hh][Rr]', '', regex=True)
    chrom_series = chrom_series.str.strip()
    
    # Define standard chromosomes
    standard_set = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                    "21", "22", "23", "24", "25",
                    "X", "Y", "M", "MT", "x", "y", "m", "mt"}
    
    # Check if in standard set
    is_standard = chrom_series.isin(standard_set)
    
    # Also check for numeric chromosomes 1-22
    # Try to convert to int and check if in range 1-22
    numeric_chroms = pd.to_numeric(chrom_series, errors='coerce')
    is_numeric_standard = (numeric_chroms >= 1) & (numeric_chroms <= 22)
    
    # Combine both checks
    result = is_standard | is_numeric_standard.fillna(False)
    
    return result.to_numpy(dtype=bool)


def parse_chain_to_segments(
    chain_path: str,
    remove_nonstandard_chromosomes: bool = False,
    remove_alternative_chromosomes: bool = False,
    remove_different_chromosomes: bool = False,
    convert_special_chromosomes: bool = False,
) -> Dict[str, Segments]:
    """
    Parse a UCSC .chain[.gz] file into per-target-chromosome segments, and build a
    disjoint best-score index for fast point liftover.
    
    This function parses chain files and creates a disjoint interval cover for each
    chromosome. Chain files often contain overlapping segments (from different chains
    or alignment blocks), so we build a non-overlapping set of intervals by selecting
    the highest-scoring segment at each position. This disjoint cover enables fast
    O(log n) coordinate lookup using binary search, as each position maps to exactly
    one segment.

    Chain format details:
      - 0-based, half-open intervals [start, end)
      - Header includes qStrand; if '-', query coordinates are on reverse strand
        (UCSC notes you may convert to forward by qStartF=qSize-qEnd, qEndF=qSize-qStart).
      - Supports both space-separated and tab-separated chain headers
      - Automatically skips comment lines (starting with `#`) and other non-chain lines
        at the beginning of the file
    
    Parameters
    ----------
    chain_path : str
        Path to UCSC chain file (can be .chain or .chain.gz).
    remove_nonstandard_chromosomes : bool, default False
        If True, filter out chains involving non-standard chromosomes (alternate contigs,
        unplaced sequences, etc.) during chain parsing. If False, include them during parsing.
        Default is False to match UCSC liftOver behavior.
    remove_alternative_chromosomes : bool, default False
        If True, filter out chains where the target chromosome (tName) is non-standard.
        This is similar to remove_nonstandard_chromosomes but specifically targets the
        output chromosome. Default is False to match UCSC liftOver behavior.
    remove_different_chromosomes : bool, default False
        If True, filter out chains where target chromosome (tName) differs from query
        chromosome (qName). This removes inter-chromosomal mappings.
        Default is False to match UCSC liftOver behavior.
    convert_special_chromosomes : bool, default False
        If True, convert special chromosomes to numeric values in qname:
        - X/x -> 23, Y/y -> 24, M/m/MT/mt -> 25
        - Also converts numeric strings (1-25) to integers
        If False, keeps chromosomes as normalized strings (X, Y, M, etc.)
    
    Returns
    -------
    Dict[str, Segments]
        Dictionary mapping normalized chromosome names to Segments objects.
        Each Segments object contains:
        - Original segments (t0, t1, q0, score, qsize, qrev, qname)
        - Disjoint best cover (bt0, bt1, bseg) for fast lookup
    """
    DEBUG_TIMING = os.environ.get('LIFTOVER_DEBUG_TIMING', '0') == '1'
    
    if DEBUG_TIMING:
        parse_t0 = time.time()
        file_read_t0 = time.time()
    
    per_t: Dict[str, List[Tuple[int, int, int, int, int, bool, str]]] = {}
    # stores: (t0, t1, q0, score, qsize, qrev, qname)

    with _open_text(chain_path) as fh:
        line = fh.readline()
        while line:
            line = line.strip()
            if not line:
                line = fh.readline()
                continue

            # Skip comment lines (starting with #) and other non-chain header lines
            # Some chain files may have comment headers at the beginning
            if line.startswith("#") or (not line.startswith("chain ") and not line.startswith("chain\t")):
                # Skip comment lines and other non-chain lines
                # This allows chain files with comment headers
                line = fh.readline()
                continue

            # Handle both space-separated and tab-separated chain headers
            # At this point, line should start with "chain " or "chain\t"

            # chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
            # split() handles both spaces and tabs
            parts = line.split()
            if len(parts) != 13:
                raise ValueError(f"Bad chain header with {len(parts)} fields: {line}")

            score = int(parts[1])
            tName = parts[2]
            # tSize = int(parts[3])  # not needed for mapping
            # tStrand = parts[4]     # typically '+'
            tStart = int(parts[5])
            # tEnd = int(parts[6])
            qName = parts[7]
            qSize = int(parts[8])
            qStrand = parts[9]
            qStart = int(parts[10])
            # qEnd = int(parts[11])
            # chain_id = parts[12]

            # Filter chains based on chromosome criteria
            should_skip = False
            
            # Filter out non-standard chromosomes (alternate contigs, unplaced sequences, etc.)
            if remove_nonstandard_chromosomes:
                if not _is_standard_chromosome(tName) or not _is_standard_chromosome(qName):
                    should_skip = True
            
            # Filter out chains where target chromosome is non-standard (alternative chromosomes)
            if not should_skip and remove_alternative_chromosomes:
                if not _is_standard_chromosome(tName):
                    should_skip = True
            
            # Filter out chains where target chromosome differs from query chromosome
            if not should_skip and remove_different_chromosomes:
                # Normalize chromosome names for comparison
                tName_norm = _normalize_chrom_name(tName)
                qName_norm = _normalize_chrom_name(qName)
                if tName_norm != qName_norm:
                    should_skip = True
            
            if should_skip:
                # Skip to next chain by reading until blank line
                while True:
                    blk = fh.readline()
                    if not blk or blk.strip() == "":
                        break
                line = fh.readline()
                continue

            qrev = (qStrand == "-")
            
            # Normalize query chromosome name during parsing
            # Remove 'chr' prefix if present (chain files typically have it)
            qName_normalized = _normalize_chrom_name(qName)
            
            # Convert special chromosomes to numeric if requested (X->23, Y->24, M->25)
            if convert_special_chromosomes:
                if qName_normalized in ("X", "x"):
                    qName_normalized = 23
                elif qName_normalized in ("Y", "y"):
                    qName_normalized = 24
                elif qName_normalized in ("M", "m", "MT", "mt"):
                    qName_normalized = 25
                else:
                    # Try to convert numeric strings to integers
                    try:
                        num_val = int(qName_normalized)
                        if 1 <= num_val <= 25:
                            qName_normalized = num_val
                    except (ValueError, TypeError):
                        pass  # Keep as string if not numeric

            # Walk blocks
            t = tStart
            q = qStart  # NOTE: on qStrand coordinates, and advances with dq
            while True:
                blk = fh.readline()
                if not blk:
                    break
                blk = blk.strip()
                if blk == "":
                    break

                nums = blk.split()
                if len(nums) == 3:
                    size = int(nums[0])
                    dt = int(nums[1])
                    dq = int(nums[2])
                    # ungapped segment
                    t0 = t
                    t1 = t + size
                    q0 = q
                    per_t.setdefault(tName, []).append((t0, t1, q0, score, qSize, qrev, qName_normalized))
                    # advance to next block
                    t += size + dt
                    q += size + dq
                elif len(nums) == 1:
                    size = int(nums[0])
                    t0 = t
                    t1 = t + size
                    q0 = q
                    per_t.setdefault(tName, []).append((t0, t1, q0, score, qSize, qrev, qName_normalized))
                    # chain ends (blank line next)
                    break
                else:
                    raise ValueError(f"Bad alignment line: {blk}")

            line = fh.readline()
    
    if DEBUG_TIMING:
        file_read_t1 = time.time()
        print(f"  Chain file reading: {file_read_t1 - file_read_t0:.3f}s")
        array_build_t0 = time.time()

    # Convert lists -> arrays and build best disjoint cover
    out: Dict[str, Segments] = {}
    for chrom, segs in per_t.items():
        # Safety check: filter non-standard chromosomes if requested
        # (Most filtering already done during parsing, but this catches edge cases)
        if remove_nonstandard_chromosomes and not _is_standard_chromosome(chrom):
            continue
            
        arr = np.array(segs, dtype=object)
        t0 = arr[:, 0].astype(np.int64)
        t1 = arr[:, 1].astype(np.int64)
        q0 = arr[:, 2].astype(np.int64)
        score = arr[:, 3].astype(np.int64)
        qsize = arr[:, 4].astype(np.int64)
        qrev = arr[:, 5].astype(bool)
        qname = arr[:, 6].astype(object)
        
        # If convert_special_chromosomes is True, qname already contains numeric values
        # Otherwise, convert any remaining string special chromosomes in the array
        if convert_special_chromosomes:
            # Convert any remaining string special chromosomes in the array
            mask_x = (qname == "X") | (qname == "x")
            mask_y = (qname == "Y") | (qname == "y")
            mask_m = (qname == "M") | (qname == "m") | (qname == "MT") | (qname == "mt")
            if np.any(mask_x):
                qname[mask_x] = 23
            if np.any(mask_y):
                qname[mask_y] = 24
            if np.any(mask_m):
                qname[mask_m] = 25

        # Sort segments by start to make sweeping deterministic
        order = np.argsort(t0, kind="mergesort")
        t0, t1, q0, score, qsize, qrev, qname = (
            t0[order], t1[order], q0[order], score[order], qsize[order], qrev[order], qname[order]
        )

        bt0, bt1, bseg = _build_best_disjoint_cover(t0, t1, score)

        # Normalize chromosome name (remove 'chr' prefix) for consistent lookup
        chrom_norm = _normalize_chrom_name(chrom)
        out[chrom_norm] = Segments(
            t0=t0, t1=t1, q0=q0, score=score, qsize=qsize, qrev=qrev, qname=qname,
            bt0=bt0, bt1=bt1, bseg=bseg
        )
    
    if DEBUG_TIMING:
        array_build_t1 = time.time()
        parse_t1 = time.time()
        print(f"  Array building & indexing: {array_build_t1 - array_build_t0:.3f}s")
        print(f"  Total chain parsing: {parse_t1 - parse_t0:.3f}s")
    
    return out


def _build_best_disjoint_cover(t0: np.ndarray, t1: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build disjoint intervals that cover (some of) the coordinate axis, choosing
    the highest-score segment among overlaps.
    
    This function solves the problem of overlapping chain segments by creating a
    non-overlapping (disjoint) set of intervals. When multiple chain segments
    overlap at the same genomic position, this algorithm selects the segment with
    the highest score to represent that region. This ensures that each position
    maps to exactly one target coordinate, making coordinate lookup fast and
    unambiguous.
    
    The algorithm uses a sweep-line approach:
    1. Create events for all segment starts and ends
    2. Sort events by position (ends before starts at same position)
    3. Sweep through positions, maintaining a heap of active segments by score
    4. At each position change, emit an interval using the highest-scoring segment
    
    Parameters
    ----------
    t0 : np.ndarray
        Start positions of segments (0-based, half-open)
    t1 : np.ndarray
        End positions of segments (0-based, half-open)
    score : np.ndarray
        Scores for each segment (higher is better)
    
    Returns
    -------
    bt0, bt1 : np.ndarray
        Disjoint intervals on target (0-based half-open)
    bseg : np.ndarray
        Segment index for each interval (indices into original t0/t1/score arrays)
    
    Example
    -------
    If segments are:
      Segment 0: [100, 200) with score 1000
      Segment 1: [150, 250) with score 2000
      Segment 2: [300, 400) with score 500
    
    The disjoint cover would be:
      [100, 150) -> Segment 0 (score 1000)
      [150, 250) -> Segment 1 (score 2000, wins overlap)
      [300, 400) -> Segment 2 (score 500)
    """
    n = t0.shape[0]
    if n == 0:
        return (np.array([], np.int64), np.array([], np.int64), np.array([], np.int32))

    # Events: (pos, typ, seg) where typ=0 for end, typ=1 for start
    # Process ends before starts at the same coordinate to avoid zero-width confusion.
    events = np.empty(2 * n, dtype=[("pos", np.int64), ("typ", np.int8), ("seg", np.int32)])
    events["pos"][0:n] = t0
    events["typ"][0:n] = 1
    events["seg"][0:n] = np.arange(n, dtype=np.int32)
    events["pos"][n:2*n] = t1
    events["typ"][n:2*n] = 0
    events["seg"][n:2*n] = np.arange(n, dtype=np.int32)

    # Sort by (pos, typ) so ends (0) come before starts (1) at same pos
    idx = np.lexsort((events["typ"], events["pos"]))
    events = events[idx]

    alive = np.zeros(n, dtype=bool)
    heap: List[Tuple[int, int]] = []  # (-score, seg)

    bt0_list: List[int] = []
    bt1_list: List[int] = []
    bseg_list: List[int] = []

    prev_pos: Optional[int] = None
    i = 0
    m = events.shape[0]

    def current_best_seg() -> Optional[int]:
        while heap and not alive[heap[0][1]]:
            heapq.heappop(heap)
        return heap[0][1] if heap else None

    while i < m:
        pos = int(events["pos"][i])

        # Emit interval from prev_pos -> pos with current best seg
        if prev_pos is not None and pos > prev_pos:
            best = current_best_seg()
            if best is not None:
                bt0_list.append(prev_pos)
                bt1_list.append(pos)
                bseg_list.append(best)

        # Process all events at this pos
        while i < m and int(events["pos"][i]) == pos:
            seg = int(events["seg"][i])
            if int(events["typ"][i]) == 1:
                alive[seg] = True
                heapq.heappush(heap, (-int(score[seg]), seg))
            else:
                alive[seg] = False
            i += 1

        prev_pos = pos

    return (
        np.array(bt0_list, dtype=np.int64),
        np.array(bt1_list, dtype=np.int64),
        np.array(bseg_list, dtype=np.int32),
    )


# ----------------------------
# Main API: liftover_df
# ----------------------------

def liftover_df(
    df: pd.DataFrame,
    chain_path: str,
    # Input column names
    chrom_col: str = "CHR",
    pos_col: str = "POS",
    # Output column names
    out_chrom_col: str = "CHR_LIFT",
    out_pos_col: str = "POS_LIFT",
    out_strand_col: str = "STRAND_LIFT",
    # Coordinate system options
    one_based_input: bool = True,
    one_based_output: bool = True,
    # Filtering options
    remove: bool = False,
    remove_unmapped: bool = False,
    remove_alternative_chromosomes: bool = False,
    remove_different_chromosomes: bool = False,
    remove_nonstandard_chromosomes: bool = False,
    # Chromosome handling options
    convert_special_chromosomes: bool = False,
    ucsc_compatible: bool = False,
) -> pd.DataFrame:
    """
    Liftover genomic coordinates in a pandas DataFrame using a UCSC chain file.
    
    This is a fast, vectorized implementation that directly parses UCSC chain files
    and performs coordinate conversion. It converts genomic coordinates from one genome
    build (e.g., hg19/GRCh37) to another (e.g., hg38/GRCh38) using UCSC chain files.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing genomic coordinates to liftover.
        Must contain columns specified by `chrom_col` and `pos_col`.
        
    chain_path : str
        Path to UCSC chain file (can be .chain or .chain.gz).
        Chain files can be downloaded from UCSC Genome Browser.
        
        Examples:
        - "/path/to/hg19ToHg38.over.chain.gz"
        - "/path/to/hg38ToHg19.over.chain"
        - Common chain files:
          * hg19ToHg38.over.chain.gz (hg19 → hg38)
          * hg38ToHg19.over.chain.gz (hg38 → hg19)
    
    chrom_col : str, default "CHR"
        Column name for chromosome in the input dataframe.
        Can be numeric (1, 2, 3..., 23, 24, 25) or string ("1", "chr1", "X", "chrX", etc.).
        Special chromosomes can be:
        - X chromosome: "X", "chrX", or 23
        - Y chromosome: "Y", "chrY", or 24
        - Mitochondrial: "M", "MT", "chrM", "chrMT", or 25
        Will be normalized internally for matching (numeric special chromosomes converted to string).
    
    pos_col : str, default "POS"
        Column name for position in the input dataframe.
        Must be integer positions.
    
    out_chrom_col : str, default "CHR_LIFT"
        Output column name for lifted chromosome.
        Chromosome names will have 'chr' prefix removed and special chromosomes
        converted to numeric values if `convert_special_chromosomes=True`:
        - Autosomal: "chr1" → "1", "chr2" → "2", etc.
        - X chromosome: "chrX" or "X" → "23"
        - Y chromosome: "chrY" or "Y" → "24"
        - Mitochondrial: "chrM", "chrMT", "M", or "MT" → "25"
    
    out_pos_col : str, default "POS_LIFT"
        Output column name for lifted position.
        Position will be in the coordinate system specified by `one_based_output`.
        Unmapped variants will have -1 or NaN.
    
    out_strand_col : str, default "STRAND_LIFT"
        Output column name for lifted strand.
        Values will be "+" (forward) or "-" (reverse).
        Unmapped variants will have None.
    
    one_based_input : bool, default True
        Whether input positions are 1-based (GWAS standard) or 0-based (BED format).
        - True: Input positions are 1-based (first base = 1)
        - False: Input positions are 0-based (first base = 0)
    
    one_based_output : bool, default True
        Whether output positions should be 1-based or 0-based.
        - True: Output positions will be 1-based (GWAS standard)
        - False: Output positions will be 0-based (BED format)
    
    remove : bool, default False
        Convenience option to filter all problematic mappings.
        When True, sets all of the following to True:
        - remove_unmapped (removes unmapped variants)
        - remove_nonstandard_chromosomes (filters non-standard chromosomes)
        - remove_alternative_chromosomes (filters alternative contigs)
        - remove_different_chromosomes (filters inter-chromosomal mappings)
        This provides a simple way to keep only clean, standard mappings with a single parameter.
    
    remove_unmapped : bool, default False
        If True, remove variants that fail to map (unmapped variants).
        If False, keep unmapped variants with out_chrom_col=None, out_pos_col=-1.
    
    convert_special_chromosomes : bool, default False
        If True, convert special chromosomes to numeric values in output:
        - X → 23, Y → 24, M/MT → 25
        If False, keep special chromosomes as strings (X, Y, M, MT).
        Default is False to match UCSC liftOver behavior.
    
    remove_alternative_chromosomes : bool, default False
        If True, filter out chains where the target chromosome is non-standard (alternate contigs,
        unplaced sequences, etc.) during chain parsing. This prevents mapping to these chromosomes.
        Filtering is done during chain loading for efficiency, so no dataframe-level filtering is needed.
        If False, allow mapping to non-standard chromosomes (matches UCSC liftOver default).
    
    remove_different_chromosomes : bool, default False
        If True, filter out chains where the target chromosome differs from the query chromosome
        during chain parsing. This prevents inter-chromosomal mappings (e.g., input is chr1 but
        output would be chr2). Filtering is done during chain loading for efficiency.
        If False, allow inter-chromosomal mappings (matches UCSC liftOver default).
    
    remove_nonstandard_chromosomes : bool, default False
        If True, filter out chains involving non-standard chromosomes (alternate contigs,
        unplaced sequences, etc.) during chain parsing. If False, include them during parsing,
        allowing mapping to these chromosomes (matches UCSC liftOver default).
    
    ucsc_compatible : bool, default False
        If True, explicitly enable UCSC liftOver-compatible behavior (same as defaults).
        This parameter is now redundant since defaults already match UCSC liftOver, but is kept
        for backward compatibility and explicit documentation. When True, ensures:
        - remove_nonstandard_chromosomes=False
        - remove_alternative_chromosomes=False
        - remove_different_chromosomes=False
    
    Returns
    -------
    pd.DataFrame
        DataFrame with lifted coordinates added as new columns.
        Original columns are preserved. New columns are added:
        - `out_chrom_col`: Lifted chromosome
        - `out_pos_col`: Lifted position  
        - `out_strand_col`: Lifted strand
        
        If remove_unmapped=True and variants are unmapped, they are excluded from output.
        If remove_unmapped=False, unmapped variants have:
        - out_chrom_col = None
        - out_pos_col = -1 or NaN
        - out_strand_col = None
        
        If remove_alternative_chromosomes=True, chains with non-standard target chromosomes
        are filtered during chain loading, so variants cannot map to these chromosomes.
        
        If remove_different_chromosomes=True, chains with different target/query chromosomes
        are filtered during chain loading, so inter-chromosomal mappings are prevented.
    
    Examples
    --------
    Basic usage:
    >>> import pandas as pd
    >>> from sumstats_liftover import liftover_df
    >>> 
    >>> df = pd.DataFrame({
    ...     'CHR': [1, 1, 2],
    ...     'POS': [725932, 725933, 100000],
    ...     'EA': ['G', 'A', 'C'],
    ...     'NEA': ['A', 'G', 'T']
    ... })
    >>> 
    >>> result = liftover_df(
    ...     df,
    ...     chain_path="/path/to/hg19ToHg38.over.chain.gz"
    ... )
    
    Custom column names:
    >>> result = liftover_df(
    ...     df,
    ...     chain_path="/path/to/hg19ToHg38.over.chain.gz",
    ...     chrom_col="Chromosome",
    ...     pos_col="BP",
    ...     out_chrom_col="CHR_hg38",
    ...     out_pos_col="POS_hg38"
    ... )
    
    Keep unmapped variants:
    >>> result = liftover_df(
    ...     df,
    ...     chain_path="/path/to/hg19ToHg38.over.chain.gz",
    ...     remove_unmapped=False
    ... )
    
    Notes
    -----
    - Chain files use 0-based, half-open intervals [start, end)
    - Chromosome name normalization handles both "1" and "chr1" formats
    - Special chromosomes are kept as strings by default (X, Y, M, MT)
      Use `convert_special_chromosomes=True` to convert to numeric (X→23, Y→24, M→25)
    - Reverse strand mappings convert coordinates to forward strand
    - Unmapped variants occur when positions fall outside alignment blocks
    - Multi-hit positions (positions that map to multiple target coordinates) are handled
      by automatically selecting the highest-scoring segment. Multi-hit is not treated
      as a failure - the best mapping is always used.
    - This implementation is faster than the original liftover for large datasets
      due to vectorized operations and optimized indexing
    - Default behavior matches UCSC liftOver (allows non-standard chromosomes, alternate contigs)
    - Use `remove=True` to filter out all problematic mappings (unmapped, non-standard, etc.)
    """
    # ========================================================================
    # Early returns and parameter setup
    # ========================================================================
    if len(df) == 0:
        return df.copy()
    
    # Handle remove convenience option
    if remove:
        remove_unmapped = True
        remove_nonstandard_chromosomes = True
        remove_alternative_chromosomes = True
        remove_different_chromosomes = True
    
    # Handle ucsc_compatible mode: explicitly set UCSC-compatible parameters
    # (Note: defaults already match UCSC, but this ensures explicit control)
    if ucsc_compatible:
        remove_nonstandard_chromosomes = False
        remove_alternative_chromosomes = False
        remove_different_chromosomes = False
    
    # Setup debug timing
    DEBUG_TIMING = os.environ.get('LIFTOVER_DEBUG_TIMING', '0') == '1'
    if DEBUG_TIMING:
        print(f"\n{'='*80}")
        print(f"LIFTOVER PERFORMANCE DEBUG - Processing {len(df):,} rows")
        print(f"{'='*80}")
        total_start = time.time()
    
    # ========================================================================
    # Parse chain file and build segment index
    # ========================================================================
    if DEBUG_TIMING:
        t0 = time.time()
    
    segs_by_chr = parse_chain_to_segments(
        chain_path,
        remove_nonstandard_chromosomes=remove_nonstandard_chromosomes,
        remove_alternative_chromosomes=remove_alternative_chromosomes,
        remove_different_chromosomes=remove_different_chromosomes,
        convert_special_chromosomes=convert_special_chromosomes
    )
    
    if DEBUG_TIMING:
        t1 = time.time()
        print(f"Chain parsing: {t1-t0:.3f}s")

    # ========================================================================
    # Prepare input data and output arrays
    # ========================================================================
    if DEBUG_TIMING:
        t0 = time.time()
    
    n = len(df)
    out_chr = np.empty(n, dtype=object)
    out_pos = np.full(n, -1, dtype=np.int64)
    out_strand = np.empty(n, dtype=object)
    
    # Extract and normalize input data
    chroms = df[chrom_col].astype(str).to_numpy()
    pos = df[pos_col].to_numpy()
    pos0 = pos.astype(np.int64) - 1 if one_based_input else pos.astype(np.int64)
    
    if DEBUG_TIMING:
        t1 = time.time()
        print(f"Data preparation: {t1-t0:.3f}s")
    
    # ========================================================================
    # Process coordinates per chromosome (vectorized lookup)
    # ========================================================================
    if DEBUG_TIMING:
        t0 = time.time()
        chrom_processing_times = {}
    
    for chrom in pd.unique(chroms):
        if DEBUG_TIMING:
            chrom_t0 = time.time()
        
        # Get indices for this chromosome
        mask = (chroms == chrom)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue
        
        # Normalize chromosome name and lookup segments
        chrom_norm = _normalize_chrom_name(chrom)
        segs = segs_by_chr.get(chrom_norm)
        
        if segs is None:
            # No mapping available for this chromosome
            out_chr[idxs] = None
            out_strand[idxs] = None
            if DEBUG_TIMING:
                chrom_t1 = time.time()
                chrom_processing_times[str(chrom)] = chrom_t1 - chrom_t0
            continue
        
        # Sort positions for efficient binary search
        p = pos0[idxs]
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]
        idxs_sorted = idxs[order]
        
        # Vectorized interval lookup using disjoint cover
        # Multi-hit positions automatically use highest-scoring segment
        j = np.searchsorted(segs.bt0, p_sorted, side="right") - 1
        ok = (j >= 0) & (p_sorted < segs.bt1[np.maximum(j, 0)])
        
        # Mark unmapped positions
        out_chr[idxs_sorted[~ok]] = None
        out_strand[idxs_sorted[~ok]] = None
        
        # Process mapped positions
        if np.any(ok):
            jj = j[ok].astype(np.int64)
            seg_idx = segs.bseg[jj].astype(np.int64)
            
            # Calculate offset from segment start
            off = p_sorted[ok] - segs.t0[seg_idx]
            
            # Initialize output position array
            qpos0 = np.empty(off.shape[0], dtype=np.int64)
            is_rev = segs.qrev[seg_idx]
            
            # Forward strand: direct mapping
            plus = ~is_rev
            if np.any(plus):
                qpos0[plus] = segs.q0[seg_idx[plus]] + off[plus]
            
            # Reverse strand: convert to forward coordinates
            # Formula: q_forward = qSize - 1 - q_reverse
            if np.any(is_rev):
                qpos0[is_rev] = (
                    segs.qsize[seg_idx[is_rev]] - 1 
                    - (segs.q0[seg_idx[is_rev]] + off[is_rev])
                )
            
            # Convert to output coordinate system (1-based or 0-based)
            qpos = qpos0 + 1 if one_based_output else qpos0
            
            # Store results
            out_pos[idxs_sorted[ok]] = qpos
            out_chr[idxs_sorted[ok]] = segs.qname[seg_idx]
            out_strand[idxs_sorted[ok]] = np.where(is_rev, "-", "+")
        
        if DEBUG_TIMING:
            chrom_t1 = time.time()
            chrom_processing_times[str(chrom)] = chrom_t1 - chrom_t0
    
    if DEBUG_TIMING:
        t1 = time.time()
        main_loop_time = t1 - t0
        print(f"Main processing loop: {main_loop_time:.3f}s")
        if chrom_processing_times:
            sorted_chroms = sorted(
                chrom_processing_times.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            print(f"  Top 5 slowest chromosomes:")
            for chrom, t in sorted_chroms[:5]:
                print(f"    Chr{chrom}: {t:.3f}s")
    
    # ========================================================================
    # Build output DataFrame
    # ========================================================================
    if DEBUG_TIMING:
        t0 = time.time()
    
    out = df.copy()
    out[out_chrom_col] = out_chr
    out[out_pos_col] = out_pos
    out[out_strand_col] = out_strand
    
    # Count and optionally remove unmapped variants
    unmapped = out[out_pos_col].isna() | (out[out_pos_col] == -1)
    unmapped_count = unmapped.sum()
    mapped_count = len(out) - unmapped_count
    
    if remove_unmapped and unmapped_count > 0:
        out = out[~unmapped].copy()
    
    if DEBUG_TIMING:
        t1 = time.time()
        print(f"Output DataFrame construction: {t1-t0:.3f}s")
        total_end = time.time()
        total_time = total_end - total_start
        print(f"{'='*80}")
        print(f"TOTAL TIME: {total_time:.3f}s ({len(df)/total_time:,.0f} rows/sec)")
        print(f"{'='*80}\n")
    
    return out


# For backward compatibility and convenience
__all__ = ['liftover_df', 'parse_chain_to_segments', 'Segments']

