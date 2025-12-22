# sumstats-liftover

Fast chain-based liftover for pandas DataFrames using UCSC chain files.

A standalone, vectorized implementation for lifting over genomic coordinates in pandas DataFrames. This library provides a fast and efficient way to convert genomic coordinates from one genome build (e.g., hg19/GRCh37) to another (e.g., hg38/GRCh38) using UCSC chain files.

## Features

- **Fast and vectorized**: Optimized for large datasets with efficient numpy-based operations
- **Standalone**: No external dependencies on UCSC tools or other liftover libraries
- **Flexible**: Supports custom column names and coordinate systems (0-based or 1-based)
- **Robust**: Handles chromosome name normalization, special chromosomes, and unmapped variants
- **Easy to use**: Simple pandas DataFrame interface

## Installation

```bash
pip install sumstats-liftover
```

Or install from source:

```bash
git clone https://github.com/yourusername/sumstats-liftover.git
cd sumstats-liftover
pip install -e .
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0

## Quick Start

```python
import pandas as pd
from sumstats_liftover import liftover_df

# Create a dataframe with genomic coordinates
df = pd.DataFrame({
    'CHR': [1, 1, 2],
    'POS': [725932, 725933, 100000],  # hg19 positions
    'EA': ['G', 'A', 'C'],
    'NEA': ['A', 'G', 'T']
})

# Perform liftover from hg19 to hg38
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    chrom_col="CHR",
    pos_col="POS"
)

print(result)
```

## Usage

### Basic Usage

```python
import pandas as pd
from sumstats_liftover import liftover_df

# Your dataframe with genomic coordinates
df = pd.DataFrame({
    'SNPID': ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C'],
    'CHR': [1, 1, 1],
    'POS': [725932, 725933, 737801],  # hg19 positions
})

# Lift over coordinates
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    chrom_col="CHR",
    pos_col="POS"
)

# Result includes original columns plus:
# - CHR_LIFT: Lifted chromosome
# - POS_LIFT: Lifted position
# - STRAND_LIFT: Strand information ("+" or "-")
```

### Custom Column Names

```python
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    chrom_col="Chromosome",
    pos_col="BP",
    out_chrom_col="CHR_hg38",
    out_pos_col="POS_hg38",
    out_strand_col="STRAND_hg38"
)
```

### Handling Unmapped Variants

By default, unmapped variants are kept with `POS_LIFT = -1` and `CHR_LIFT = None`. To remove them:

```python
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    remove_unmapped=True
)
```

### Coordinate Systems

The library supports both 0-based (BED format) and 1-based (GWAS standard) coordinates:

```python
# For 0-based input coordinates
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    one_based_input=False,
    one_based_output=False
)
```

## Chain Files

UCSC chain files can be downloaded from the [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/downloads.html). Common chain files include:

- `hg19ToHg38.over.chain.gz` - Convert from hg19 to hg38
- `hg38ToHg19.over.chain.gz` - Convert from hg38 to hg19
- `hg18ToHg19.over.chain.gz` - Convert from hg18 to hg19

## API Reference

### `liftover_df()`

Main function for lifting over genomic coordinates in a pandas DataFrame.

**Parameters:**

- `df` (pd.DataFrame): DataFrame containing genomic coordinates
- `chain_path` (str): Path to UCSC chain file (`.chain` or `.chain.gz`)
- `chrom_col` (str, default="CHR"): Column name for chromosome
- `pos_col` (str, default="POS"): Column name for position
- `out_chrom_col` (str, default="CHR_LIFT"): Output column name for lifted chromosome
- `out_pos_col` (str, default="POS_LIFT"): Output column name for lifted position
- `out_strand_col` (str, default="STRAND_LIFT"): Output column name for lifted strand
- `one_based_input` (bool, default=True): Whether input positions are 1-based
- `one_based_output` (bool, default=True): Whether output positions should be 1-based
- `remove_unmapped` (bool, default=False): Remove variants that fail to map
- `convert_special_chromosomes` (bool, default=True): Convert X→23, Y→24, M/MT→25

**Returns:**

- `pd.DataFrame`: DataFrame with lifted coordinates added as new columns

**Example:**

```python
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    chrom_col="CHR",
    pos_col="POS"
)
```

## Chromosome Name Handling

The library automatically handles various chromosome name formats:

- **Input formats**: `1`, `chr1`, `X`, `chrX`, `23` (for X), `24` (for Y), `25` (for M/MT)
- **Output format**: By default, special chromosomes are converted to numeric values:
  - X → 23
  - Y → 24
  - M/MT → 25

To keep special chromosomes as strings:

```python
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    convert_special_chromosomes=False
)
```

## Testing

Run the test suite:

```bash
pytest test_liftover_df.py -v
```

## Example

See `example.py` for a complete example demonstrating liftover with a real dataset:

```bash
python example.py
```

## How It Works

### Building Disjoint Intervals

UCSC chain files often contain overlapping segments from different alignment chains. To enable fast and unambiguous coordinate lookup, this library builds a **disjoint interval cover** that selects the highest-scoring segment at each position.

**The Problem:**
- Chain files contain multiple segments that can overlap at the same genomic positions
- Each position needs to map to exactly one target coordinate
- We need to choose which segment to use when overlaps occur

**The Solution:**
The library uses a sweep-line algorithm to build non-overlapping (disjoint) intervals:

1. **Parse segments**: Extract all alignment segments from the chain file
2. **Build disjoint cover**: For overlapping regions, select the segment with the highest score
3. **Create index**: Build a sorted array of disjoint intervals for O(log n) lookup

**Example:**
If we have overlapping segments:
- Segment A: [100, 200) with score 1000
- Segment B: [150, 250) with score 2000
- Segment C: [300, 400) with score 500

The disjoint cover becomes:
- [100, 150) → Segment A (only A covers this region)
- [150, 250) → Segment B (B has higher score than A in overlap)
- [300, 400) → Segment C (no overlap)

This ensures each position maps to exactly one target coordinate, enabling fast binary search lookup.

## Performance

This implementation is optimized for large datasets and uses vectorized numpy operations for fast coordinate conversion. The disjoint interval index enables O(log n) coordinate lookup, making it typically faster than the original UCSC liftover tool for batch processing of large DataFrames.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{sumstats-liftover,
  title = {sumstats-liftover: Fast chain-based liftover for pandas DataFrames},
  author = {Yunye He},
  year = {2025},
  url = {https://github.com/yourusername/sumstats-liftover}
}
```

## Links

- [GitHub Repository](https://github.com/yourusername/sumstats-liftover)
- [Issue Tracker](https://github.com/yourusername/sumstats-liftover/issues)
- [UCSC Genome Browser](https://genome.ucsc.edu/)

