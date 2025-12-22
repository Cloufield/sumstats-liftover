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


def parse_chain_to_segments(chain_path: str) -> Dict[str, Segments]:
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
      - header includes qStrand; if '-', query coordinates are on reverse strand
        (UCSC notes you may convert to forward by qStartF=qSize-qEnd, qEndF=qSize-qStart).
    
    Returns
    -------
    Dict[str, Segments]
        Dictionary mapping normalized chromosome names to Segments objects.
        Each Segments object contains:
        - Original segments (t0, t1, q0, score, qsize, qrev, qname)
        - Disjoint best cover (bt0, bt1, bseg) for fast lookup
    """
    per_t: Dict[str, List[Tuple[int, int, int, int, int, bool, str]]] = {}
    # stores: (t0, t1, q0, score, qsize, qrev, qname)

    with _open_text(chain_path) as fh:
        line = fh.readline()
        while line:
            line = line.strip()
            if not line:
                line = fh.readline()
                continue

            if not line.startswith("chain "):
                raise ValueError(f"Unexpected line (expected chain header): {line[:80]}")

            # chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
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

            # Filter out non-standard chromosomes (alternate contigs, unplaced sequences, etc.)
            # Skip chains that involve non-standard chromosomes
            if not _is_standard_chromosome(tName) or not _is_standard_chromosome(qName):
                # Skip to next chain by reading until blank line
                while True:
                    blk = fh.readline()
                    if not blk or blk.strip() == "":
                        break
                line = fh.readline()
                continue

            qrev = (qStrand == "-")

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
                    per_t.setdefault(tName, []).append((t0, t1, q0, score, qSize, qrev, qName))
                    # advance to next block
                    t += size + dt
                    q += size + dq
                elif len(nums) == 1:
                    size = int(nums[0])
                    t0 = t
                    t1 = t + size
                    q0 = q
                    per_t.setdefault(tName, []).append((t0, t1, q0, score, qSize, qrev, qName))
                    # chain ends (blank line next)
                    break
                else:
                    raise ValueError(f"Bad alignment line: {blk}")

            line = fh.readline()

    # Convert lists -> arrays and build best disjoint cover
    out: Dict[str, Segments] = {}
    for chrom, segs in per_t.items():
        # Filter out non-standard chromosomes (alternate contigs, unplaced sequences, etc.)
        if not _is_standard_chromosome(chrom):
            continue
            
        arr = np.array(segs, dtype=object)
        t0 = arr[:, 0].astype(np.int64)
        t1 = arr[:, 1].astype(np.int64)
        q0 = arr[:, 2].astype(np.int64)
        score = arr[:, 3].astype(np.int64)
        qsize = arr[:, 4].astype(np.int64)
        qrev = arr[:, 5].astype(bool)
        qname = arr[:, 6].astype(object)

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
    chrom_col: str = "CHR",
    pos_col: str = "POS",
    out_chrom_col: str = "CHR_LIFT",
    out_pos_col: str = "POS_LIFT",
    out_strand_col: str = "STRAND_LIFT",
    one_based_input: bool = True,
    one_based_output: bool = True,
    remove_unmapped: bool = False,
    convert_special_chromosomes: bool = True,
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
    
    remove_unmapped : bool, default False
        If True, remove variants that fail to map (unmapped variants).
        If False, keep unmapped variants with out_chrom_col=None, out_pos_col=-1.
    
    convert_special_chromosomes : bool, default True
        If True, convert special chromosomes to numeric values in output:
        - X → 23, Y → 24, M/MT → 25
        If False, keep special chromosomes as strings (X, Y, M, MT).
    
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
    - Special chromosomes are converted to numeric values in output by default:
      X → 23, Y → 24, M/MT → 25
    - Reverse strand mappings convert coordinates to forward strand
    - Unmapped variants occur when positions fall outside alignment blocks
    - This implementation is faster than the original liftover for large datasets
      due to vectorized operations and optimized indexing
    - Non-standard chromosomes (alternate contigs) are filtered out during chain parsing
    """
    if len(df) == 0:
        return df.copy()
    
    segs_by_chr = parse_chain_to_segments(chain_path)

    # Prepare output arrays
    n = len(df)
    out_chr = np.empty(n, dtype=object)
    out_pos = np.full(n, -1, dtype=np.int64)
    out_strand = np.empty(n, dtype=object)

    chroms = df[chrom_col].astype(str).to_numpy()
    pos = df[pos_col].to_numpy()
    pos0 = pos.astype(np.int64) - 1 if one_based_input else pos.astype(np.int64)

    # Process per chromosome for speed
    # (Using stable grouping by chromosome values present in df)
    for chrom in pd.unique(chroms):
        mask = (chroms == chrom)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue

        # Normalize chromosome name for lookup (remove 'chr' prefix if present)
        chrom_norm = _normalize_chrom_name(chrom)
        segs = segs_by_chr.get(chrom_norm)
        if segs is None:
            # no mapping for this contig
            out_chr[idxs] = None
            out_strand[idxs] = None
            continue

        p = pos0[idxs]
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]
        idxs_sorted = idxs[order]

        # Vectorized interval lookup on best-disjoint cover
        j = np.searchsorted(segs.bt0, p_sorted, side="right") - 1
        ok = (j >= 0) & (p_sorted < segs.bt1[np.maximum(j, 0)])

        # Fill unmapped
        out_chr[idxs_sorted[~ok]] = None
        out_strand[idxs_sorted[~ok]] = None

        if np.any(ok):
            jj = j[ok].astype(np.int64)
            seg_idx = segs.bseg[jj].astype(np.int64)

            # offset from original segment start (not bt0)
            off = p_sorted[ok] - segs.t0[seg_idx]

            qpos0 = np.empty(off.shape[0], dtype=np.int64)
            is_rev = segs.qrev[seg_idx]

            # '+' strand
            plus = ~is_rev
            if np.any(plus):
                qpos0[plus] = segs.q0[seg_idx[plus]] + off[plus]

            # '-' strand: convert to forward coordinate base by base
            # q_forward = qSize - 1 - (q_reverse)
            if np.any(is_rev):
                qpos0[is_rev] = segs.qsize[seg_idx[is_rev]] - 1 - (segs.q0[seg_idx[is_rev]] + off[is_rev])

            qpos = qpos0 + 1 if one_based_output else qpos0

            out_pos[idxs_sorted[ok]] = qpos
            # Normalize query chromosome names and filter out non-standard chromosomes
            qname_values = segs.qname[seg_idx]
            # Normalize qName (remove 'chr' prefix) for consistency
            qname_normalized = np.array([_normalize_chrom_name(str(qn)) for qn in qname_values])
            out_chr[idxs_sorted[ok]] = qname_normalized
            out_strand[idxs_sorted[ok]] = np.where(is_rev, "-", "+")

    out = df.copy()
    out[out_chrom_col] = out_chr
    out[out_pos_col] = out_pos
    out[out_strand_col] = out_strand
    
    # Count unmapped variants
    unmapped = out[out_pos_col].isna() | (out[out_pos_col] == -1)
    unmapped_count = unmapped.sum()
    mapped_count = len(out) - unmapped_count
    
    # Remove unmapped variants if requested
    if remove_unmapped and unmapped_count > 0:
        out = out[~unmapped].copy()
    
    # Update chromosome format if needed (strip 'chr' prefix and convert special chromosomes)
    if convert_special_chromosomes and out_chrom_col in out.columns:
        # Convert chromosome names to match expected format
        # First, strip 'chr' prefix
        out[out_chrom_col] = out[out_chrom_col].astype(str).str.replace("^chr", "", regex=True)
        # Only convert special chromosomes for standard chromosomes (filter out alternate contigs)
        # Check which values are standard chromosomes before converting
        is_standard = out[out_chrom_col].apply(_is_standard_chromosome)
        if is_standard.any():
            # Convert special chromosomes to numeric: X→23, Y→24, M/MT→25
            # Only apply to standard chromosomes
            standard_mask = is_standard
            out.loc[standard_mask, out_chrom_col] = out.loc[standard_mask, out_chrom_col].replace({
                "X": "23", "x": "23", "chrX": "23", "chrx": "23",
                "Y": "24", "y": "24", "chrY": "24", "chry": "24",
                "M": "25", "m": "25", "MT": "25", "mt": "25", "chrM": "25", "chrm": "25", "chrMT": "25", "chrmt": "25"
            })
            
            # Convert to numeric if possible (for special chromosomes 23, 24, 25)
            # Use errors='coerce' to handle non-numeric chromosome names (alternate contigs, etc.)
            # Only convert standard chromosomes that can be safely converted
            try:
                # First, identify which values are standard chromosomes and can be converted
                chrom_series = out[out_chrom_col].astype(str)
                # Check which are standard chromosomes
                is_standard = chrom_series.apply(_is_standard_chromosome)
                
                # Only try to convert standard chromosomes
                if is_standard.any():
                    # For standard chromosomes, try to convert to numeric
                    # This handles: "1"-"22" -> 1-22, "23"->23, "24"->24, "25"->25
                    # Create a copy for conversion
                    chrom_to_convert = chrom_series.copy()
                    # Only convert standard chromosomes
                    numeric_chrom = pd.to_numeric(chrom_to_convert, errors='coerce')
                    # Only update values that are standard AND successfully converted (not NaN)
                    valid_mask = is_standard & (~numeric_chrom.isna())
                    if valid_mask.any():
                        out.loc[valid_mask, out_chrom_col] = numeric_chrom[valid_mask].astype('Int64')
                # Non-standard chromosomes (alternate contigs, etc.) remain as strings
            except Exception:
                # If conversion fails entirely, keep original values as strings
                pass
    
    return out


# For backward compatibility and convenience
__all__ = ['liftover_df', 'parse_chain_to_segments', 'Segments']

