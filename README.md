# sumstats-liftover

Fast chain-based liftover for pandas DataFrames using UCSC chain files.

A standalone, vectorized implementation for lifting over genomic coordinates in pandas DataFrames. This library provides a fast and efficient way to convert genomic coordinates from one genome build (e.g., hg19/GRCh37) to another (e.g., hg38/GRCh38) using UCSC chain files.

**Note**: This module is part of [GWASLab](https://cloufield.github.io/gwaslab/), a comprehensive Python package for processing and visualizing GWAS summary statistics.

## Features

- **Fast**: ~1.2M rows/second throughput, 24-25x faster than UCSC liftOver
- **Built-in chain files**: Includes commonly used chain files (hg19↔hg38, hg18→hg19)
- **Standalone**: No external dependencies on UCSC tools
- **Flexible**: Custom column names, 0-based/1-based coordinates
- **Robust**: Handles chromosome normalization, special chromosomes, unmapped variants
- **Accurate**: 100% agreement with UCSC liftOver for standard chromosomes

## Installation

```bash
pip install sumstats-liftover
```

**Requirements**: Python >= 3.8, numpy >= 1.20.0, pandas >= 1.3.0

## Quick Start

```python
import pandas as pd
from sumstats_liftover import liftover_df, get_chain_path

# Create dataframe with genomic coordinates
df = pd.DataFrame({
    'CHR': [1, 1, 2],
    'POS': [725932, 725933, 100000],  # hg19 positions
    'EA': ['G', 'A', 'C'],
    'NEA': ['A', 'G', 'T']
})

# Perform liftover using built-in chain file
result = liftover_df(
    df,
    chain_path=get_chain_path("hg19ToHg38"),
    chrom_col="CHR",
    pos_col="POS"
)

print(result[['CHR', 'POS', 'CHR_LIFT', 'POS_LIFT', 'STRAND_LIFT']])
```

## Usage

### Built-in Chain Files

The package includes commonly used chain files:

```python
from sumstats_liftover import get_chain_path, list_chain_files

# List available chain files
list_chain_files()
# {'hg19ToHg38': 'Convert from hg19/GRCh37 to hg38/GRCh38',
#  'hg38ToHg19': 'Convert from hg38/GRCh38 to hg19/GRCh37',
#  'hg18ToHg19': 'Convert from hg18 to hg19/GRCh37'}

# Use built-in chain file
result = liftover_df(df, chain_path=get_chain_path("hg19ToHg38"))
```

### Custom Chain Files

Use your own chain files by providing the path:

```python
result = liftover_df(df, chain_path="/path/to/custom.chain.gz")
```

UCSC chain files: [Download](https://hgdownload.soe.ucsc.edu/downloads.html)

**Note**: The parser supports both space-separated and tab-separated chain files, and automatically handles comment headers (lines starting with `#`) at the beginning of chain files.

### Filtering Options

**Default behavior** matches UCSC liftOver (allows non-standard chromosomes, alternate contigs, inter-chromosomal mappings).

**Filter problematic mappings:**

```python
# Remove all problematic mappings with one parameter
result = liftover_df(df, chain_path=chain_path, remove=True)

# Or control individually
result = liftover_df(
    df,
    chain_path=chain_path,
    remove_unmapped=True,                    # Remove unmapped variants
    remove_nonstandard_chromosomes=True,    # Filter non-standard chromosomes
    remove_alternative_chromosomes=True,     # Filter alternative contigs
    remove_different_chromosomes=True        # Filter inter-chromosomal mappings
)
```

### Coordinate Systems

```python
# 0-based input/output (BED format)
result = liftover_df(df, chain_path=chain_path, 
                     one_based_input=False, one_based_output=False)

# 1-based input/output (GWAS standard, default)
result = liftover_df(df, chain_path=chain_path,
                     one_based_input=True, one_based_output=True)
```

### Custom Column Names

```python
result = liftover_df(
    df,
    chain_path=chain_path,
    chrom_col="Chromosome",
    pos_col="BP",
    out_chrom_col="CHR_hg38",
    out_pos_col="POS_hg38"
)
```

### Special Chromosomes

By default, special chromosomes (X, Y, M) are kept as strings. Convert to numeric:

```python
result = liftover_df(df, chain_path=chain_path, 
                     convert_special_chromosomes=True)  # X→23, Y→24, M→25
```

## API Reference

### `liftover_df()`

Main function for lifting over genomic coordinates.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pd.DataFrame | - | DataFrame with genomic coordinates |
| `chain_path` | str | - | Path to UCSC chain file |
| `chrom_col` | str | `"CHR"` | Input chromosome column name |
| `pos_col` | str | `"POS"` | Input position column name |
| `out_chrom_col` | str | `"CHR_LIFT"` | Output chromosome column name |
| `out_pos_col` | str | `"POS_LIFT"` | Output position column name |
| `out_strand_col` | str | `"STRAND_LIFT"` | Output strand column name |
| `one_based_input` | bool | `True` | Whether input is 1-based |
| `one_based_output` | bool | `True` | Whether output should be 1-based |
| `remove` | bool | `False` | Remove all problematic mappings (convenience option) |
| `remove_unmapped` | bool | `False` | Remove unmapped variants |
| `remove_nonstandard_chromosomes` | bool | `False` | Filter non-standard chromosomes |
| `remove_alternative_chromosomes` | bool | `False` | Filter alternative contigs |
| `remove_different_chromosomes` | bool | `False` | Filter inter-chromosomal mappings |
| `convert_special_chromosomes` | bool | `False` | Convert X→23, Y→24, M→25 |
| `ucsc_compatible` | bool | `False` | Explicit UCSC-compatible mode (redundant with defaults) |

**Returns:** `pd.DataFrame` with lifted coordinates added as new columns.

### Chain File Functions

- `get_chain_path(name)` - Get path to built-in chain file
- `list_chain_files()` - List all available built-in chain files
- `get_chain_info(name)` - Get information about a chain file

## Performance

### Benchmarks

| Dataset Size | Time | Throughput | Memory |
|--------------|------|------------|--------|
| 1,000 rows | ~0.19s | ~5,200 rows/s | < 10 MB |
| 10,000 rows | ~0.19s | ~54,000 rows/s | < 20 MB |
| 1,000,000 rows | ~0.84s | ~1,190,000 rows/s | ~200 MB |
| 30,000,000 rows | ~24s | ~1,250,000 rows/s | ~2 GB |

**Key characteristics:**
- Consistent ~1.2M rows/second throughput across all sizes
- Linear scaling with dataset size
- Memory efficient: ~60-80 KB per row

### Comparison with UCSC liftOver

| Tool | Throughput | Time (1M) | Time (30M) | Speed |
|------|------------|-----------|------------|-------|
| sumstats-liftover | ~1.2M rows/s | 0.84s | ~24s | **24-25x faster** |
| UCSC liftOver | ~48.6K rows/s | 20.58s | ~617s | Baseline |

**Accuracy:** 100% agreement with UCSC liftOver for standard chromosome mappings (tested on 1M variants).

## How It Works

The library builds a **disjoint interval cover** from UCSC chain files by selecting the highest-scoring segment at each position when overlaps occur. This enables O(log n) coordinate lookup using binary search.

**Algorithm:**
1. Parse all alignment segments from chain file
2. Build disjoint cover: for overlaps, select highest-scoring segment
3. Create sorted index for fast binary search lookup

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run performance tests
pytest tests/test_performance.py -v -s

# Run accuracy tests
pytest tests/test_variant_types.py -v
```

See `example.py` for usage examples.

## License

**Package**: MIT License (see LICENSE file)

**UCSC Chain Files**: Built-in chain files are proprietary to The Regents of the University of California:
- Free for Independent Researchers and Nonprofit Organizations (non-commercial use)
- Commercial use requires UCSC license
- [EULA](https://genome.ucsc.edu/license/EULA.pdf) | [Licensing](https://genome.ucsc.edu/license/)

Users are responsible for ensuring compliance with UCSC EULA.

## Citation

**GWASLab** (main package):
```bibtex
@article{he2023gwaslab,
  title = {GWASLab: a Python package for processing and visualizing GWAS summary statistics},
  author = {He, Yunye and Koido, Masaru and Shimmori, Yoichi and Kamatani, Yoichiro},
  year = {2023},
  journal = {Jxiv},
  doi = {10.51094/jxiv.370}
}
```

**sumstats-liftover**:
```bibtex
@software{sumstats-liftover,
  title = {sumstats-liftover: Fast chain-based liftover for pandas DataFrames},
  author = {He, Yunye},
  year = {2024},
  url = {https://github.com/yourusername/sumstats-liftover},
  note = {Module of GWASLab}
}
```

## Links

- [GWASLab](https://cloufield.github.io/gwaslab/) - Main package
- [GitHub Repository](https://github.com/yourusername/sumstats-liftover)
- [UCSC Genome Browser](https://genome.ucsc.edu/)
