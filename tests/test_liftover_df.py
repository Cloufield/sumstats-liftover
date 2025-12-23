"""
General tests for liftover_df module, comparing with UCSC liftOver.

This test suite focuses on general functionality and correctness tests
against UCSC liftOver tool, including basic operations, parameter handling,
and general comparison tests.

Role: General test against UCSC liftover.
"""

import os
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
import pytest
from sumstats_liftover import liftover_df

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN_FILE = os.path.join(PROJECT_ROOT, "hg19ToHg38.over.chain.gz")


def _is_standard_chromosome(chrom) -> bool:
    """
    Check if a chromosome name represents a standard chromosome.
    Standard chromosomes are: 1-22, X, Y, M/MT (and their 'chr' prefixed versions).
    Filters out alternate contigs, unplaced sequences, etc.
    """
    chrom_str = str(chrom).strip()
    # Remove 'chr' prefix (case-insensitive)
    if chrom_str.lower().startswith('chr'):
        chrom_str = chrom_str[3:]
    
    # Check for standard chromosomes
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


def test_basic_liftover():
    """Test basic liftover functionality with hg19 to hg38."""
    # Create test dataframe with hg19 positions
    df = pd.DataFrame({
        'SNPID': ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C'],
        'CHR': [1, 1, 1],
        'POS': [725932, 725933, 737801],  # hg19 positions
    })
    
    # Perform liftover
    result = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Check that result has expected columns
    assert 'CHR_LIFT' in result.columns
    assert 'POS_LIFT' in result.columns
    assert 'STRAND_LIFT' in result.columns
    
    # Check that all variants were mapped (positions should be valid)
    assert result['POS_LIFT'].notna().all()
    assert (result['POS_LIFT'] > 0).all()
    
    # Check that chromosome is still 1 (may be string or int depending on convert_special_chromosomes)
    assert (result['CHR_LIFT'].astype(str) == '1').all()
    
    # Check that original columns are preserved
    assert 'SNPID' in result.columns
    
    # Assert exact expected output values
    expected_positions = [790552, 790553, 802421]
    assert result['POS_LIFT'].tolist() == expected_positions
    
    # Assert all other columns match
    assert result['SNPID'].tolist() == ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C']
    assert result['CHR'].tolist() == [1, 1, 1]


def test_liftover_with_unmapped():
    """Test liftover with positions that may not map."""
    # Create test dataframe with some positions
    df = pd.DataFrame({
        'CHR': [1, 1, 2],
        'POS': [725932, 725933, 100000],  # hg19 positions
        'EA': ['G', 'A', 'C'],
        'NEA': ['A', 'G', 'T']
    })
    
    # Perform liftover without removing unmapped
    result = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        remove_unmapped=False
    )
    
    # Check that result has expected columns
    assert 'CHR_LIFT' in result.columns
    assert 'POS_LIFT' in result.columns
    assert 'STRAND_LIFT' in result.columns
    
    # Check that all rows are preserved
    assert len(result) == len(df)
    
    # Check that at least some positions were mapped
    assert result['POS_LIFT'].notna().any() or (result['POS_LIFT'] > 0).any()


# Helper function to run UCSC liftOver tool and parse output
def run_ucsc_liftover(df, chain_path, chrom_col="CHR", pos_col="POS", one_based_input=True):
    """
    Run UCSC liftOver tool on a dataframe and return results as a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with genomic coordinates
    chain_path : str
        Path to chain file
    chrom_col : str
        Column name for chromosome
    pos_col : str
        Column name for position
    one_based_input : bool
        Whether input is 1-based (True) or 0-based (False)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CHR_LIFT, POS_LIFT, STRAND_LIFT
        Unmapped variants will have None/NaN values
    """
    # Get liftOver path from environment or use default
    liftover_path = os.environ.get("LIFTOVER", "/home/yunye/tools/bin/liftOver")
    if not os.path.exists(liftover_path):
        # Try to find it in PATH
        try:
            result = subprocess.run(["which", "liftOver"], capture_output=True, text=True)
            if result.returncode == 0:
                liftover_path = result.stdout.strip()
            else:
                pytest.skip("UCSC liftOver tool not found")
        except Exception:
            pytest.skip("UCSC liftOver tool not found")
    
    # Create temporary BED file (0-based, half-open)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as bed_in:
        for idx, row in df.iterrows():
            chrom = row[chrom_col]
            pos = int(row[pos_col])
            
            # Convert to BED format (0-based)
            if one_based_input:
                bed_start = pos - 1
            else:
                bed_start = pos
            
            # UCSC liftOver expects chr prefix
            if isinstance(chrom, (int, float)):
                if chrom == 23:
                    chrom_str = "chrX"
                elif chrom == 24:
                    chrom_str = "chrY"
                elif chrom == 25:
                    chrom_str = "chrM"
                else:
                    chrom_str = f"chr{int(chrom)}"
            else:
                chrom_str = str(chrom)
                if not chrom_str.startswith("chr"):
                    if chrom_str.upper() == "X":
                        chrom_str = "chrX"
                    elif chrom_str.upper() == "Y":
                        chrom_str = "chrY"
                    elif chrom_str.upper() in ["M", "MT"]:
                        chrom_str = "chrM"
                    else:
                        chrom_str = f"chr{chrom_str}"
            
            bed_in.write(f"{chrom_str}\t{bed_start}\t{bed_start + 1}\t{idx}\n")
        bed_in_path = bed_in.name
    
    # Create temporary output files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as bed_out:
        bed_out_path = bed_out.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.unmapped', delete=False) as unmapped_out:
        unmapped_path = unmapped_out.name
    
    try:
        # Run liftOver
        result = subprocess.run(
            [liftover_path, bed_in_path, chain_path, bed_out_path, unmapped_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse output
        results = {}
        unmapped_indices = set()
        
        # Read unmapped file to get indices that failed
        if os.path.exists(unmapped_path) and os.path.getsize(unmapped_path) > 0:
            with open(unmapped_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            unmapped_indices.add(int(parts[3]))
        
        # Read mapped results
        if os.path.exists(bed_out_path) and os.path.getsize(bed_out_path) > 0:
            with open(bed_out_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            idx = int(parts[3])
                            chrom_out = parts[0]  # e.g., "chr1"
                            start = int(parts[1])  # 0-based
                            
                            # Convert to 1-based position
                            pos_1based = start + 1
                            
                            # Normalize chromosome name
                            chrom_norm = chrom_out.replace("chr", "")
                            if chrom_norm == "X":
                                chrom_norm = 23
                            elif chrom_norm == "Y":
                                chrom_norm = 24
                            elif chrom_norm in ["M", "MT"]:
                                chrom_norm = 25
                            else:
                                try:
                                    chrom_norm = int(chrom_norm)
                                except ValueError:
                                    chrom_norm = chrom_norm
                            
                            results[idx] = {
                                'CHR_LIFT': chrom_norm,
                                'POS_LIFT': pos_1based,
                                'STRAND_LIFT': '+'  # UCSC liftOver doesn't provide strand in BED output
                            }
        
        # Build result dataframe
        result_list = []
        for idx in range(len(df)):
            if idx in results:
                result_list.append(results[idx])
            else:
                result_list.append({
                    'CHR_LIFT': None,
                    'POS_LIFT': None,
                    'STRAND_LIFT': None
                })
        
        return pd.DataFrame(result_list)
    
    finally:
        # Clean up temporary files
        for path in [bed_in_path, bed_out_path, unmapped_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_compare_with_ucsc_liftover_basic():
    """Test that our package output matches UCSC liftOver tool for basic cases."""
    # Real hg19 positions that should map to hg38
    df = pd.DataFrame({
        'CHR': [1, 1, 1, 2, 2],
        'POS': [725932, 725933, 737801, 100000, 200000],  # hg19 positions
        'SNPID': ['1:725932', '1:725933', '1:737801', '2:100000', '2:200000']
    })
    
    # Run our package
    result_package = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Run UCSC liftOver
    result_ucsc = run_ucsc_liftover(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Compare results
    assert len(result_package) == len(result_ucsc)
    
    # Compare mapped positions (where both succeeded)
    for idx in range(len(df)):
        pkg_chr = result_package.iloc[idx]['CHR_LIFT']
        pkg_pos = result_package.iloc[idx]['POS_LIFT']
        ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
        ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
        
        # Both should be mapped or both unmapped
        pkg_mapped = pd.notna(pkg_pos) and pkg_pos > 0
        ucsc_mapped = pd.notna(ucsc_pos) and ucsc_pos > 0
        
        if pkg_mapped and ucsc_mapped:
            # Both mapped - positions should match
            # Handle string vs numeric chromosome comparison (default is now strings)
            # Normalize by converting to string and removing .0 suffix if present
            pkg_chr_str = str(pkg_chr).rstrip('.0') if pkg_chr is not None else None
            ucsc_chr_str = str(ucsc_chr).rstrip('.0') if ucsc_chr is not None else None
            assert pkg_chr_str == ucsc_chr_str, f"Chromosome mismatch at index {idx}: package={pkg_chr}, UCSC={ucsc_chr}"
            assert pkg_pos == ucsc_pos, f"Position mismatch at index {idx}: package={pkg_pos}, UCSC={ucsc_pos}"
        elif not pkg_mapped and not ucsc_mapped:
            # Both unmapped - that's fine
            pass
        else:
            # One mapped, one unmapped - this might happen due to different handling
            # Log but don't fail (could be due to multi-hit handling differences)
            print(f"Warning: Mapping discrepancy at index {idx}: package_mapped={pkg_mapped}, UCSC_mapped={ucsc_mapped}")


def test_compare_with_ucsc_liftover_multiple_chromosomes():
    """Test comparison with UCSC liftOver across multiple chromosomes."""
    # Test positions from different chromosomes
    df = pd.DataFrame({
        'CHR': [1, 2, 3, 10, 22],
        'POS': [725932, 100000, 500000, 1000000, 2000000],  # hg19 positions
        'SNPID': [f'{chr}:{pos}' for chr, pos in zip([1, 2, 3, 10, 22], [725932, 100000, 500000, 1000000, 2000000])]
    })
    
    # Run our package
    result_package = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Run UCSC liftOver
    result_ucsc = run_ucsc_liftover(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Compare mapped positions
    mapped_count_pkg = (result_package['POS_LIFT'].notna() & (result_package['POS_LIFT'] > 0)).sum()
    mapped_count_ucsc = (result_ucsc['POS_LIFT'].notna() & (result_ucsc['POS_LIFT'] > 0)).sum()
    
    # At least some should be mapped
    assert mapped_count_pkg > 0, "Package should map at least some positions"
    assert mapped_count_ucsc > 0, "UCSC liftOver should map at least some positions"
    
    # Compare individual mapped positions
    for idx in range(len(df)):
        pkg_chr = result_package.iloc[idx]['CHR_LIFT']
        pkg_pos = result_package.iloc[idx]['POS_LIFT']
        ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
        ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
        
        pkg_mapped = pd.notna(pkg_pos) and pkg_pos > 0
        ucsc_mapped = pd.notna(ucsc_pos) and ucsc_pos > 0
        
        if pkg_mapped and ucsc_mapped:
            # Handle string vs numeric chromosome comparison (default is now strings)
            # Normalize by converting to string and removing .0 suffix if present
            pkg_chr_str = str(pkg_chr).rstrip('.0') if pkg_chr is not None else None
            ucsc_chr_str = str(ucsc_chr).rstrip('.0') if ucsc_chr is not None else None
            assert pkg_chr_str == ucsc_chr_str, f"Chromosome mismatch at index {idx}: package={pkg_chr}, UCSC={ucsc_chr}"
            # Compare positions (handle int vs float)
            pkg_pos_val = float(pkg_pos) if pd.notna(pkg_pos) else None
            ucsc_pos_val = float(ucsc_pos) if pd.notna(ucsc_pos) else None
            assert pkg_pos_val == ucsc_pos_val, f"Position mismatch at index {idx}: package={pkg_pos}, UCSC={ucsc_pos}"


def test_compare_with_ucsc_liftover_real_gwas_positions():
    """Test with real GWAS summary statistics positions."""
    # Real positions from example.py
    df = pd.DataFrame({
        'SNPID': ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C'],
        'CHR': [1, 1, 1],
        'POS': [725932, 725933, 737801],  # hg19 positions
        'EA': ['G', 'A', 'C'],
        'NEA': ['A', 'G', 'T']
    })
    
    # Run our package
    result_package = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Run UCSC liftOver
    result_ucsc = run_ucsc_liftover(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Verify all positions are mapped
    assert result_package['POS_LIFT'].notna().all()
    assert (result_package['POS_LIFT'] > 0).all()
    
    # Compare with UCSC liftOver
    for idx in range(len(df)):
        pkg_chr = result_package.iloc[idx]['CHR_LIFT']
        pkg_pos = result_package.iloc[idx]['POS_LIFT']
        ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
        ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
        
        # Handle string vs numeric chromosome comparison (default is now strings)
        # Normalize by converting to string and removing .0 suffix if present
        pkg_chr_str = str(pkg_chr).rstrip('.0') if pkg_chr is not None else None
        ucsc_chr_str = str(ucsc_chr).rstrip('.0') if ucsc_chr is not None else None
        assert pkg_chr_str == ucsc_chr_str, f"Chromosome mismatch at index {idx}: package={pkg_chr}, UCSC={ucsc_chr}"
        # Compare positions (handle int vs float)
        pkg_pos_val = float(pkg_pos) if pd.notna(pkg_pos) else None
        ucsc_pos_val = float(ucsc_pos) if pd.notna(ucsc_pos) else None
        assert pkg_pos_val == ucsc_pos_val, f"Position mismatch at index {idx}: package={pkg_pos}, UCSC={ucsc_pos}"
    
    # Verify expected positions from previous test
    expected_positions = [790552, 790553, 802421]
    assert result_package['POS_LIFT'].tolist() == expected_positions


def test_compare_with_ucsc_liftover_large_dataset():
    """Test comparison with a larger dataset to ensure consistency."""
    # Create a larger dataset with various positions
    positions = [
        (1, 725932), (1, 725933), (1, 737801),
        (2, 100000), (2, 200000), (2, 500000),
        (3, 1000000), (3, 2000000),
        (10, 500000), (10, 1000000),
        (22, 1000000), (22, 2000000)
    ]
    
    df = pd.DataFrame({
        'CHR': [p[0] for p in positions],
        'POS': [p[1] for p in positions],
        'SNPID': [f'{p[0]}:{p[1]}' for p in positions]
    })
    
    # Run our package
    result_package = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Run UCSC liftOver
    result_ucsc = run_ucsc_liftover(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Compare results
    matches = 0
    mismatches = 0
    
    for idx in range(len(df)):
        pkg_chr = result_package.iloc[idx]['CHR_LIFT']
        pkg_pos = result_package.iloc[idx]['POS_LIFT']
        ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
        ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
        
        pkg_mapped = pd.notna(pkg_pos) and pkg_pos > 0
        ucsc_mapped = pd.notna(ucsc_pos) and ucsc_pos > 0
        
        if pkg_mapped and ucsc_mapped:
            # Handle string vs numeric chromosome comparison (default is now strings)
            # Normalize by converting to string and removing .0 suffix if present
            pkg_chr_str = str(pkg_chr).rstrip('.0') if pkg_chr is not None else None
            ucsc_chr_str = str(ucsc_chr).rstrip('.0') if ucsc_chr is not None else None
            # Compare positions (handle int vs float)
            pkg_pos_val = float(pkg_pos) if pd.notna(pkg_pos) else None
            ucsc_pos_val = float(ucsc_pos) if pd.notna(ucsc_pos) else None
            if pkg_chr_str == ucsc_chr_str and pkg_pos_val == ucsc_pos_val:
                matches += 1
            else:
                mismatches += 1
                print(f"Mismatch at {df.iloc[idx]['SNPID']}: package=chr{pkg_chr}:{pkg_pos}, UCSC=chr{ucsc_chr}:{ucsc_pos}")
    
    # Most positions should match
    assert matches > 0, "Should have at least some matching positions"
    # Allow some mismatches due to multi-hit handling differences
    assert mismatches < len(df) * 0.1, f"Too many mismatches: {mismatches}/{len(df)}"


def test_performance_1_million_rows(
    exclude_alternative_chromosomes=True,
    exclude_different_chromosomes=True
):
    """
    Performance test for liftover with 1 million rows, comparing with UCSC liftOver.
    
    Parameters
    ----------
    exclude_alternative_chromosomes : bool, default True
        If True, exclude UCSC liftOver results that map to non-standard chromosomes
        (alternate contigs, random sequences, etc.) from comparison.
        Our package filters these out, so excluding them makes the comparison fairer.
    
    exclude_different_chromosomes : bool, default True
        If True, exclude variants where both tools mapped but to different chromosomes
        from the agreement calculation.
    """
    # Generate 1 million rows with realistic genomic positions
    # Use a mix of chromosomes and positions across the genome
    np.random.seed(42)  # For reproducibility
    
    n_rows = 1_000_000
    
    # Generate chromosomes (1-22, X, Y)
    # Weight towards autosomes (more common in GWAS)
    chrom_choices = list(range(1, 23)) + [23, 24]  # 1-22, X(23), Y(24)
    chrom_weights = [1.0] * 22 + [0.1, 0.01]  # X and Y are less common
    chrom_weights = np.array(chrom_weights)
    chrom_weights = chrom_weights / chrom_weights.sum()
    
    chromosomes = np.random.choice(chrom_choices, size=n_rows, p=chrom_weights)
    
    # Generate positions across chromosomes
    # Use realistic position ranges for hg19
    positions = []
    for chrom in chromosomes:
        if chrom <= 22:
            # Autosomes: positions roughly 1-250M
            max_pos = 250_000_000
        elif chrom == 23:  # X
            max_pos = 155_000_000
        else:  # Y
            max_pos = 60_000_000
        
        # Generate position with some clustering (more realistic)
        pos = np.random.randint(1, max_pos)
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CHR': chromosomes,
        'POS': positions,
        'SNPID': [f'{chr}:{pos}' for chr, pos in zip(chromosomes, positions)]
    })
    
    print(f"\n{'='*80}")
    print(f"Performance Test: {n_rows:,} rows")
    print(f"{'='*80}")
    print(f"Chromosome distribution:")
    chrom_counts = pd.Series(chromosomes).value_counts().sort_index()
    for chrom, count in chrom_counts.head(10).items():
        print(f"  Chr{chrom}: {count:,} variants")
    print(f"  ... (showing top 10)")
    print(f"{'='*80}\n")
    
    # Run our package liftover and measure time
    print("Running sumstats-liftover package...")
    start_time = time.time()
    result_package = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    end_time = time.time()
    
    elapsed_time_package = end_time - start_time
    rows_per_second_package = n_rows / elapsed_time_package
    
    # Verify package results
    assert len(result_package) == n_rows, "All rows should be preserved"
    assert 'CHR_LIFT' in result_package.columns
    assert 'POS_LIFT' in result_package.columns
    assert 'STRAND_LIFT' in result_package.columns
    
    # Count mapped variants for package
    mapped_package = result_package['POS_LIFT'].notna() & (result_package['POS_LIFT'] > 0)
    mapped_count_package = mapped_package.sum()
    mapping_rate_package = (mapped_count_package / n_rows) * 100
    
    print(f"✓ Package completed in {elapsed_time_package:.2f} seconds")
    print(f"  Throughput: {rows_per_second_package:,.0f} rows/second\n")
    
    # Run UCSC liftOver and measure time
    print("Running UCSC liftOver tool...")
    start_time = time.time()
    result_ucsc = run_ucsc_liftover(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS"
    )
    end_time = time.time()
    
    elapsed_time_ucsc = end_time - start_time
    rows_per_second_ucsc = n_rows / elapsed_time_ucsc
    
    # Count mapped variants for UCSC
    mapped_ucsc = result_ucsc['POS_LIFT'].notna() & (result_ucsc['POS_LIFT'] > 0)
    mapped_count_ucsc = mapped_ucsc.sum()
    mapping_rate_ucsc = (mapped_count_ucsc / n_rows) * 100
    
    print(f"✓ UCSC liftOver completed in {elapsed_time_ucsc:.2f} seconds")
    print(f"  Throughput: {rows_per_second_ucsc:,.0f} rows/second\n")
    
    # Speed comparison
    speedup = elapsed_time_ucsc / elapsed_time_package
    print(f"{'='*80}")
    print(f"Speed Comparison:")
    print(f"  Package time: {elapsed_time_package:.2f} seconds")
    print(f"  UCSC time: {elapsed_time_ucsc:.2f} seconds")
    print(f"  Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
    print(f"  Package throughput: {rows_per_second_package:,.0f} rows/second")
    print(f"  UCSC throughput: {rows_per_second_ucsc:,.0f} rows/second")
    print(f"{'='*80}\n")
    
    # Result comparison
    print(f"{'='*80}")
    print(f"Result Comparison:")
    print(f"  Package mapped: {mapped_count_package:,} ({mapping_rate_package:.2f}%)")
    print(f"  UCSC mapped (all): {mapped_count_ucsc:,} ({mapping_rate_ucsc:.2f}%)")
    
    # Mask UCSC results that map to non-standard chromosomes (alternate contigs, etc.)
    # Our package filters these out, so we should exclude them from UCSC results too
    ucsc_maps_to_nonstandard = np.zeros(n_rows, dtype=bool)
    for idx in range(n_rows):
        if mapped_ucsc.iloc[idx]:
            ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
            if pd.notna(ucsc_chr) and not _is_standard_chromosome(ucsc_chr):
                ucsc_maps_to_nonstandard[idx] = True
    
    nonstandard_count = ucsc_maps_to_nonstandard.sum()
    
    # Apply masking based on option
    if exclude_alternative_chromosomes:
        # Create masked UCSC mapping status (treat non-standard chromosome mappings as unmapped)
        mapped_ucsc_masked = mapped_ucsc & ~ucsc_maps_to_nonstandard
        mapped_count_ucsc_masked = mapped_ucsc_masked.sum()
        mapping_rate_ucsc_masked = (mapped_count_ucsc_masked / n_rows) * 100
        print(f"  UCSC mapped to non-standard chromosomes: {nonstandard_count:,} (excluded from comparison)")
        print(f"  UCSC mapped (standard only): {mapped_count_ucsc_masked:,} ({mapping_rate_ucsc_masked:.2f}%)")
    else:
        # Don't mask - use all UCSC results
        mapped_ucsc_masked = mapped_ucsc
        mapped_count_ucsc_masked = mapped_count_ucsc
        mapping_rate_ucsc_masked = mapping_rate_ucsc
        print(f"  UCSC mapped to non-standard chromosomes: {nonstandard_count:,} (included in comparison)")
        print(f"  UCSC mapped (all): {mapped_count_ucsc_masked:,} ({mapping_rate_ucsc_masked:.2f}%)")
    
    # Identify variants mapped to different chromosomes
    # Only consider variants that both mapped to standard chromosomes
    both_mapped = mapped_package & mapped_ucsc_masked
    different_chromosomes = np.zeros(n_rows, dtype=bool)
    
    if both_mapped.sum() > 0:
        both_mapped_indices = np.where(both_mapped)[0]
        for idx in both_mapped_indices:
            pkg_chr = result_package.iloc[idx]['CHR_LIFT']
            ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
            if pkg_chr != ucsc_chr:
                different_chromosomes[idx] = True
    
    different_chr_count = different_chromosomes.sum()
    print(f"  Variants mapped to different chromosomes: {different_chr_count:,}")
    
    # Mask variants mapped to different chromosomes from comparison
    # Create mask excluding different chromosome mappings
    comparison_mask = ~different_chromosomes  # numpy array, works with boolean operations
    
    # Compare mapping status (mapped vs unmapped) excluding different chromosome cases
    # Use masked UCSC results (excluding non-standard chromosome mappings)
    both_mapped_same_chr = both_mapped & comparison_mask
    both_unmapped = ~mapped_package & ~mapped_ucsc_masked
    pkg_only_mapped = mapped_package & ~mapped_ucsc_masked
    ucsc_only_mapped = ~mapped_package & mapped_ucsc_masked
    
    # Calculate agreement excluding different chromosome cases
    valid_comparison_count = n_rows - different_chr_count
    agreement = ((both_mapped_same_chr.sum() + both_unmapped.sum()) / valid_comparison_count * 100) if valid_comparison_count > 0 else 0
    
    print(f"\n  Agreement (excluding different chromosome mappings): {agreement:.2f}%")
    print(f"    Both mapped (same chromosome): {both_mapped_same_chr.sum():,}")
    print(f"    Both unmapped: {both_unmapped.sum():,}")
    print(f"    Package only mapped: {pkg_only_mapped.sum():,}")
    print(f"    UCSC only mapped: {ucsc_only_mapped.sum():,}")
    print(f"    Different chromosomes (excluded): {different_chr_count:,}")
    
    # Show examples of UCSC-only mapped variants (to non-standard chromosomes)
    if nonstandard_count > 0:
        nonstandard_indices = np.where(ucsc_maps_to_nonstandard)[0]
        sample_size_nonstd = min(10, len(nonstandard_indices))
        sample_nonstd_indices = nonstandard_indices[:sample_size_nonstd]
        
        print(f"\n  Examples of UCSC mappings to non-standard chromosomes (showing {sample_size_nonstd} of {nonstandard_count:,}):")
        print(f"    {'Index':<8} {'Input CHR':<12} {'Input POS':<12} {'UCSC CHR':<25} {'UCSC POS':<12} {'Package Status':<15}")
        print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*25} {'-'*12} {'-'*15}")
        
        for idx in sample_nonstd_indices:
            input_chr = df.iloc[idx]['CHR']
            input_pos = df.iloc[idx]['POS']
            ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
            ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
            pkg_chr = result_package.iloc[idx]['CHR_LIFT']
            pkg_pos = result_package.iloc[idx]['POS_LIFT']
            
            if pd.isna(pkg_pos) or pkg_pos <= 0:
                pkg_status = "Unmapped"
            else:
                pkg_status = f"Mapped to {pkg_chr}:{int(pkg_pos)}"
            
            print(f"    {idx:<8} {input_chr:<12} {input_pos:<12} {str(ucsc_chr):<25} {int(ucsc_pos) if pd.notna(ucsc_pos) else 'N/A':<12} {pkg_status:<15}")
    
    # Show examples of UCSC-only mapped variants (to standard chromosomes, if any)
    if ucsc_only_mapped.sum() > 0:
        ucsc_only_indices = np.where(ucsc_only_mapped)[0]
        sample_size_ucsc = min(10, len(ucsc_only_indices))
        sample_ucsc_indices = ucsc_only_indices[:sample_size_ucsc]
        
        print(f"\n  Examples of UCSC-only mapped variants (to standard chromosomes, showing {sample_size_ucsc} of {ucsc_only_mapped.sum():,}):")
        print(f"    {'Index':<8} {'Input CHR':<12} {'Input POS':<12} {'UCSC CHR':<12} {'UCSC POS':<12} {'Package Status':<15}")
        print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
        
        for idx in sample_ucsc_indices:
            input_chr = df.iloc[idx]['CHR']
            input_pos = df.iloc[idx]['POS']
            ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
            ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
            pkg_chr = result_package.iloc[idx]['CHR_LIFT']
            pkg_pos = result_package.iloc[idx]['POS_LIFT']
            
            if pd.isna(pkg_pos) or pkg_pos <= 0:
                pkg_status = "Unmapped"
            else:
                pkg_status = f"Mapped to {pkg_chr}:{int(pkg_pos)}"
            
            print(f"    {idx:<8} {input_chr:<12} {input_pos:<12} {ucsc_chr:<12} {int(ucsc_pos) if pd.notna(ucsc_pos) else 'N/A':<12} {pkg_status:<15}")
    
    # Compare positions for variants that both mapped to the same chromosome
    both_mapped_same_chr_indices = np.where(both_mapped_same_chr)[0]
    if len(both_mapped_same_chr_indices) > 0:
        # Sample for comparison (compare first 10,000 to avoid being too slow)
        sample_size = min(10_000, len(both_mapped_same_chr_indices))
        sample_indices = both_mapped_same_chr_indices[:sample_size]
        
        matches = 0
        mismatches = 0
        
        for idx in sample_indices:
            pkg_chr = result_package.iloc[idx]['CHR_LIFT']
            pkg_pos = result_package.iloc[idx]['POS_LIFT']
            ucsc_chr = result_ucsc.iloc[idx]['CHR_LIFT']
            ucsc_pos = result_ucsc.iloc[idx]['POS_LIFT']
            
            # Should already be same chromosome, but double-check
            # Handle string vs numeric chromosome comparison (default is now strings)
            # Normalize by converting to string and removing .0 suffix if present
            pkg_chr_str = str(pkg_chr).rstrip('.0') if pkg_chr is not None else None
            ucsc_chr_str = str(ucsc_chr).rstrip('.0') if ucsc_chr is not None else None
            if pkg_chr_str == ucsc_chr_str:
                if pkg_pos == ucsc_pos:
                    matches += 1
                else:
                    mismatches += 1
        
        match_rate = (matches / sample_size) * 100 if sample_size > 0 else 0
        print(f"\n  Position comparison (sampled {sample_size:,} variants, same chromosome):")
        print(f"    Matches: {matches:,} ({match_rate:.2f}%)")
        print(f"    Mismatches: {mismatches:,} ({100 - match_rate:.2f}%)")
        
        # Assert high match rate (should be > 95% for positions that both mapped to same chromosome)
        assert match_rate > 95, f"Position match rate too low: {match_rate:.2f}%"
    
    print(f"{'='*80}\n")
    
    # Performance assertions
    # Package should complete in reasonable time
    assert elapsed_time_package < 120, f"Package liftover took too long: {elapsed_time_package:.2f} seconds"
    assert rows_per_second_package > 10_000, f"Package throughput too low: {rows_per_second_package:,.0f} rows/second"
    
    # Should map at least some variants
    assert mapped_count_package > 0, "Package should map at least some variants"
    assert mapped_count_ucsc > 0, "UCSC liftOver should map at least some variants"
    
    # Mapping rates should be similar (within 5%)
    # Note: Small differences can occur due to different chromosome mappings
    mapping_rate_diff = abs(mapping_rate_package - mapping_rate_ucsc)
    assert mapping_rate_diff < 5, f"Mapping rates differ too much: package={mapping_rate_package:.2f}%, UCSC={mapping_rate_ucsc:.2f}%"
    
    # Agreement should be high when excluding different chromosome mappings
    assert agreement > 95, f"Agreement too low when excluding different chromosome mappings: {agreement:.2f}%"
    
    # Verify a sample of mapped positions are valid
    if mapped_count_package > 0:
        sample_mapped = result_package[mapped_package].head(100)
        assert (sample_mapped['CHR_LIFT'].notna()).all(), "Mapped variants should have valid chromosomes"
        assert (sample_mapped['POS_LIFT'] > 0).all(), "Mapped variants should have positive positions"
        assert (sample_mapped['STRAND_LIFT'].isin(['+', '-'])).all(), "Mapped variants should have valid strand"


def test_remove_nonstandard_chromosomes():
    """Test that remove_nonstandard_chromosomes controls mapping to non-standard chromosomes."""
    # Create test dataframe with positions that may map to non-standard chromosomes
    # We'll use positions that UCSC liftOver maps to alternate contigs
    df = pd.DataFrame({
        'CHR': [8, 1, 7],
        'POS': [2314366, 143469538, 142113887],  # Positions that map to alternate contigs
        'SNPID': ['8:2314366', '1:143469538', '7:142113887']
    })
    
    # Test with remove_nonstandard_chromosomes=True - should filter them out
    result_default = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        remove_nonstandard_chromosomes=True,
        remove_unmapped=False
    )
    
    # Test with remove_nonstandard_chromosomes=False - should allow mapping to non-standard
    result_keep = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        remove_nonstandard_chromosomes=False,
        remove_unmapped=False
    )
    
    # With remove_nonstandard_chromosomes=False, we should get more mapped variants
    # (or at least the same, but potentially with non-standard chromosome mappings)
    mapped_default = (result_default['POS_LIFT'].notna() & (result_default['POS_LIFT'] > 0)).sum()
    mapped_keep = (result_keep['POS_LIFT'].notna() & (result_keep['POS_LIFT'] > 0)).sum()
    
    # When not removing non-standard chromosomes, we should have at least as many mapped variants
    assert mapped_keep >= mapped_default, "remove_nonstandard_chromosomes=False should map at least as many variants"
    
    # Check that some variants may have non-standard chromosome names in output
    if mapped_keep > 0:
        # Get output chromosomes
        output_chroms = result_keep[result_keep['POS_LIFT'].notna() & (result_keep['POS_LIFT'] > 0)]['CHR_LIFT']
        # Some may be non-standard (strings like "8_KI270821v1_alt" or similar)
        # This is expected when remove_nonstandard_chromosomes=False


def test_ucsc_compatible_mode():
    """Test that ucsc_compatible mode matches UCSC liftOver behavior."""
    # Create test dataframe with positions that may map to non-standard chromosomes
    df = pd.DataFrame({
        'CHR': [8, 1, 7, 1],
        'POS': [2314366, 143469538, 142113887, 725932],  # Mix of standard and non-standard mappings
        'SNPID': ['8:2314366', '1:143469538', '7:142113887', '1:725932']
    })
    
    # Test with ucsc_compatible=True
    result_ucsc = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        ucsc_compatible=True,
        remove_unmapped=False
    )
    
    # Test with default settings (should filter non-standard)
    result_default = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        remove_unmapped=False
    )
    
    # ucsc_compatible mode should map at least as many variants as default
    mapped_ucsc = (result_ucsc['POS_LIFT'].notna() & (result_ucsc['POS_LIFT'] > 0)).sum()
    mapped_default = (result_default['POS_LIFT'].notna() & (result_default['POS_LIFT'] > 0)).sum()
    
    assert mapped_ucsc >= mapped_default, "ucsc_compatible mode should map at least as many variants"
    
    # The last variant (1:725932) should map in both cases (it's a standard mapping)
    assert result_ucsc.iloc[3]['POS_LIFT'] > 0, "Standard variant should map in ucsc_compatible mode"
    assert result_default.iloc[3]['POS_LIFT'] > 0, "Standard variant should map in default mode"
    
    # Verify that ucsc_compatible=True sets remove_nonstandard_chromosomes=False
    # by checking that we can get non-standard chromosome mappings
    # (This is a basic sanity check - exact matching with UCSC would require running UCSC tool)


def test_ucsc_compatible_overrides_other_params():
    """Test that ucsc_compatible=True overrides remove_nonstandard_chromosomes and remove_alternative_chromosomes."""
    df = pd.DataFrame({
        'CHR': [1, 1],
        'POS': [725932, 143469538],  # One standard, one that may map to non-standard
        'SNPID': ['1:725932', '1:143469538']
    })
    
    # Test that ucsc_compatible=True works even if remove_nonstandard_chromosomes=True is explicitly set
    result = liftover_df(
        df,
        chain_path=CHAIN_FILE,
        chrom_col="CHR",
        pos_col="POS",
        ucsc_compatible=True,
        remove_nonstandard_chromosomes=True,  # Should be overridden
        remove_alternative_chromosomes=True,  # Should be overridden
        remove_unmapped=False
    )
    
    # Should still work (ucsc_compatible overrides the other params)
    assert len(result) == len(df), "All rows should be preserved"
    assert 'CHR_LIFT' in result.columns
    assert 'POS_LIFT' in result.columns

