# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-12-25

### Added
- Support for tab-separated chain files (in addition to space-separated)
  - Automatically detects and handles both formats
  - Tested with grch38-chm13v2.chain and other tab-separated chain files
- Support for chain files with comment headers
  - Automatically skips comment lines (starting with `#`) and other non-chain header lines
  - Allows chain files with metadata or comments at the beginning

### Fixed
- Chain file parser now correctly handles chain headers with tabs instead of spaces
- Improved compatibility with chain files from various sources (UCSC, T2T, etc.)
- Chain file parser now gracefully handles comment headers and other non-chain lines

## [1.0.0] - 2025-12-23

### Added
- Built-in chain files for common genome build conversions:
  - `hg19ToHg38` - Convert from hg19/GRCh37 to hg38/GRCh38
  - `hg38ToHg19` - Convert from hg38/GRCh38 to hg19/GRCh37
  - `hg18ToHg19` - Convert from hg18 to hg19/GRCh37
- Helper functions for accessing built-in chain files:
  - `get_chain_path()` - Get path to built-in chain file
  - `list_chain_files()` - List all available built-in chain files
  - `get_chain_info()` - Get information about a chain file
- Comprehensive test suite with 40 tests covering:
  - General functionality tests (`test_liftover_df.py`)
  - Performance benchmarks (`test_performance.py`)
  - Accuracy and corner case tests (`test_variant_types.py`)
- Performance benchmarking across multiple dataset sizes (1K to 50M rows)
- Extensive documentation in README with performance metrics
- Support for filtering options:
  - `remove`: Convenience option to filter all problematic mappings
  - `remove_unmapped`: Filter unmapped variants
  - `remove_nonstandard_chromosomes`: Filter non-standard chromosomes
  - `remove_alternative_chromosomes`: Filter alternative contigs
  - `remove_different_chromosomes`: Filter inter-chromosomal mappings
- `ucsc_compatible` mode for explicit UCSC liftOver compatibility
- `convert_special_chromosomes` option to convert X/Y/M to numeric (23/24/25)
- Support for various chromosome formats (numeric, string, with/without 'chr' prefix)
- Memory-efficient processing with optimized chain file parsing
- Vectorized operations for high-throughput processing (~1.2M rows/second)

### Changed
- Default behavior now matches UCSC liftOver:
  - All filtering options default to `False` (allows non-standard chromosomes, alternate contigs)
  - Special chromosomes kept as strings by default (X, Y, M)
  - Inter-chromosomal mappings allowed by default
- Improved performance with consistent throughput across all dataset sizes
- Enhanced code organization and documentation

### Fixed
- Consistent handling of chromosome name normalization
- Proper handling of boundary positions and edge cases
- Accurate coordinate system conversion (0-based vs 1-based)
- Multi-hit position handling (selects highest-scoring segment)

### Performance
- Maintains ~1.2M rows/second throughput for datasets from 1K to 50M rows
- 24-25x faster than UCSC liftOver tool
- Memory efficient: ~60-80 KB per row
- Linear scaling with dataset size

### Documentation
- Comprehensive README with:
  - Performance benchmarks table
  - API reference
  - Usage examples
  - Performance optimization tips
  - Comparison with UCSC liftOver

### Testing
- 100% accuracy agreement with UCSC liftOver for standard chromosome mappings
- Comprehensive test coverage including:
  - Basic functionality
  - Unmapped variants
  - Multiple chromosomes
  - Special chromosome formats
  - Non-standard chromosome mappings
  - Inter-chromosomal mappings
  - Boundary positions
  - Corner cases (empty dataframes, single variants, duplicates, etc.)
  - Performance benchmarks
  - Memory efficiency tests

## [0.2.0] - Previous version

Initial development version with basic liftover functionality.

## [0.1.0] - Initial version

Initial release with core liftover functionality.

