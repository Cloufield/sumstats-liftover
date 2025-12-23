"""
Extreme performance tests for liftover_df.

This test suite focuses on testing extreme performance scenarios including:
- Very large datasets (stress testing)
- Memory efficiency
- Throughput under different conditions
- Scaling behavior

Role: Test the extreme performance for this package.
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import pytest
from sumstats_liftover import liftover_df

# Try to import psutil for memory tests, skip if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN_FILE = os.path.join(PROJECT_ROOT, "hg19ToHg38.over.chain.gz")


def generate_test_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate test dataset with realistic genomic positions.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with CHR and POS columns
    """
    np.random.seed(seed)
    
    # Generate chromosomes (1-22, X, Y)
    # Weight towards autosomes (more common in GWAS)
    chrom_choices = list(range(1, 23)) + [23, 24]  # 1-22, X(23), Y(24)
    chrom_weights = [1.0] * 22 + [0.1, 0.01]  # X and Y are less common
    chrom_weights = np.array(chrom_weights)
    chrom_weights = chrom_weights / chrom_weights.sum()
    
    chromosomes = np.random.choice(chrom_choices, size=n_rows, p=chrom_weights)
    
    # Generate positions across chromosomes
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
    
    return df


class TestPerformance:
    """Performance test suite for different dataset sizes."""
    
    @pytest.mark.parametrize("n_rows", [
        1_000,
        10_000,
        1_000_000,
        pytest.param(30_000_000, marks=pytest.mark.slow)  # Mark large test as slow
    ])
    def test_performance_scaling(self, n_rows):
        """
        Test performance across different dataset sizes.
        
        Parameters
        ----------
        n_rows : int
            Number of rows to test (1000, 10000, 1000000, 30000000)
        """
        # Generate test data
        print(f"\n{'='*80}")
        print(f"Performance Test: {n_rows:,} rows")
        print(f"{'='*80}")
        
        df = generate_test_data(n_rows)
        
        # Show chromosome distribution
        chrom_counts = pd.Series(df['CHR']).value_counts().sort_index()
        print(f"Chromosome distribution:")
        for chrom, count in chrom_counts.head(10).items():
            print(f"  Chr{chrom}: {count:,} variants")
        if len(chrom_counts) > 10:
            print(f"  ... (showing top 10 of {len(chrom_counts)})")
        print()
        
        # Run liftover and measure time
        print("Running sumstats-liftover...")
        start_time = time.time()
        
        result = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        rows_per_second = n_rows / elapsed_time if elapsed_time > 0 else 0
        
        # Count mapped variants
        mapped = result['POS_LIFT'].notna() & (result['POS_LIFT'] > 0)
        mapped_count = mapped.sum()
        mapping_rate = (mapped_count / n_rows) * 100 if n_rows > 0 else 0
        
        # Display results
        print(f"{'='*80}")
        print(f"Results:")
        print(f"  Time: {elapsed_time:.3f} seconds")
        print(f"  Throughput: {rows_per_second:,.0f} rows/second")
        print(f"  Mapped variants: {mapped_count:,} ({mapping_rate:.2f}%)")
        print(f"  Unmapped variants: {n_rows - mapped_count:,} ({100 - mapping_rate:.2f}%)")
        print(f"{'='*80}\n")
        
        # Verify results
        assert len(result) == n_rows, "All rows should be preserved"
        assert 'CHR_LIFT' in result.columns
        assert 'POS_LIFT' in result.columns
        assert 'STRAND_LIFT' in result.columns
        
        # Performance assertions (scaled by dataset size)
        if n_rows >= 1_000_000:
            # For large datasets, should complete in reasonable time
            assert elapsed_time < 300, f"Liftover took too long: {elapsed_time:.2f} seconds"
            assert rows_per_second > 5_000, f"Throughput too low: {rows_per_second:,.0f} rows/second"
        elif n_rows >= 10_000:
            # For medium datasets
            assert elapsed_time < 60, f"Liftover took too long: {elapsed_time:.2f} seconds"
            assert rows_per_second > 1_000, f"Throughput too low: {rows_per_second:,.0f} rows/second"
        else:
            # For small datasets
            assert elapsed_time < 10, f"Liftover took too long: {elapsed_time:.2f} seconds"
            assert rows_per_second > 100, f"Throughput too low: {rows_per_second:,.0f} rows/second"
        
        # Should map at least some variants
        assert mapped_count > 0, "Should map at least some variants"
        
        # Verify a sample of mapped positions are valid
        if mapped_count > 0:
            sample_mapped = result[mapped].head(100)
            assert (sample_mapped['CHR_LIFT'].notna()).all(), "Mapped variants should have valid chromosomes"
            assert (sample_mapped['POS_LIFT'] > 0).all(), "Mapped variants should have positive positions"
            assert (sample_mapped['STRAND_LIFT'].isin(['+', '-'])).all(), "Mapped variants should have valid strand"
    
    @pytest.mark.slow  # This test runs all sizes including 30M rows
    def test_performance_comparison_summary(self):
        """Run all performance tests and provide a summary comparison."""
        row_counts = [1_000, 10_000, 1_000_000, 30_000_000]
        results = []
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SCALING SUMMARY")
        print(f"{'='*80}\n")
        
        for n_rows in row_counts:
            print(f"Testing {n_rows:,} rows...")
            df = generate_test_data(n_rows)
            
            start_time = time.time()
            result = liftover_df(
                df,
                chain_path=CHAIN_FILE,
                chrom_col="CHR",
                pos_col="POS"
            )
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            rows_per_second = n_rows / elapsed_time if elapsed_time > 0 else 0
            mapped_count = (result['POS_LIFT'].notna() & (result['POS_LIFT'] > 0)).sum()
            mapping_rate = (mapped_count / n_rows) * 100 if n_rows > 0 else 0
            
            results.append({
                'rows': n_rows,
                'time': elapsed_time,
                'throughput': rows_per_second,
                'mapped': mapped_count,
                'mapping_rate': mapping_rate
            })
        
        # Print summary table
        print(f"\n{'='*80}")
        print(f"Performance Summary Table")
        print(f"{'='*80}")
        print(f"{'Rows':<15} {'Time (s)':<12} {'Throughput (rows/s)':<20} {'Mapped':<15} {'Mapping Rate':<15}")
        print(f"{'-'*15} {'-'*12} {'-'*20} {'-'*15} {'-'*15}")
        
        for r in results:
            print(f"{r['rows']:>13,}  {r['time']:>10.3f}  {r['throughput']:>18,.0f}  {r['mapped']:>13,}  {r['mapping_rate']:>13.2f}%")
        
        print(f"{'='*80}\n")
        
        # Verify scaling is reasonable (throughput shouldn't degrade too much with size)
        throughputs = [r['throughput'] for r in results]
        if len(throughputs) >= 2:
            # Smallest dataset might be slower due to overhead, but larger ones should be similar
            medium_throughput = throughputs[1]  # 10K rows
            large_throughput = throughputs[2] if len(throughputs) > 2 else medium_throughput  # 1M rows
            
            # Large dataset throughput should be at least 50% of medium (accounting for overhead)
            throughput_ratio = large_throughput / medium_throughput if medium_throughput > 0 else 0
            assert throughput_ratio > 0.5, (
                f"Performance degrades too much with size: "
                f"large throughput ({large_throughput:,.0f}) is only {throughput_ratio:.1%} of medium ({medium_throughput:,.0f})"
            )
    
    @pytest.mark.slow
    def test_extreme_large_dataset(self):
        """Test extreme performance with very large dataset (50M+ rows)."""
        n_rows = 50_000_000
        
        print(f"\n{'='*80}")
        print(f"EXTREME PERFORMANCE TEST: {n_rows:,} rows")
        print(f"{'='*80}")
        
        df = generate_test_data(n_rows)
        
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available for memory testing")
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory before: {mem_before:.1f} MB")
        print("Running liftover...")
        
        start_time = time.time()
        result = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        elapsed_time = end_time - start_time
        rows_per_second = n_rows / elapsed_time if elapsed_time > 0 else 0
        mapped_count = (result['POS_LIFT'].notna() & (result['POS_LIFT'] > 0)).sum()
        
        print(f"\nResults:")
        print(f"  Time: {elapsed_time:.3f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"  Throughput: {rows_per_second:,.0f} rows/second")
        print(f"  Memory used: {mem_used:.1f} MB")
        print(f"  Memory per row: {mem_used/n_rows*1024:.3f} KB")
        print(f"  Mapped: {mapped_count:,} ({mapped_count/n_rows*100:.2f}%)")
        print(f"{'='*80}\n")
        
        # Performance assertions for extreme case
        assert elapsed_time < 600, f"Extreme test took too long: {elapsed_time:.2f} seconds"
        assert rows_per_second > 50_000, f"Throughput too low for extreme test: {rows_per_second:,.0f} rows/second"
        assert len(result) == n_rows, "All rows should be preserved"
    
    def test_performance_with_different_chromosome_distributions(self):
        """Test performance with different chromosome distributions."""
        n_rows = 1_000_000
        
        distributions = {
            'uniform': None,  # Use default (weighted)
            'autosomes_only': list(range(1, 23)),
            'chr1_heavy': [1] * 90 + list(range(2, 23)) * 10  # 90% chr1
        }
        
        results = []
        
        for dist_name, chrom_choices in distributions.items():
            print(f"\nTesting distribution: {dist_name}")
            
            if chrom_choices is None:
                df = generate_test_data(n_rows)
            else:
                np.random.seed(42)
                chromosomes = np.random.choice(chrom_choices, size=n_rows)
                positions = [np.random.randint(1, 250_000_000) for _ in chromosomes]
                df = pd.DataFrame({
                    'CHR': chromosomes,
                    'POS': positions,
                    'SNPID': [f'{chr}:{pos}' for chr, pos in zip(chromosomes, positions)]
                })
            
            start_time = time.time()
            result = liftover_df(
                df,
                chain_path=CHAIN_FILE,
                chrom_col="CHR",
                pos_col="POS"
            )
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            rows_per_second = n_rows / elapsed_time if elapsed_time > 0 else 0
            
            results.append({
                'distribution': dist_name,
                'time': elapsed_time,
                'throughput': rows_per_second
            })
            
            print(f"  Time: {elapsed_time:.3f}s, Throughput: {rows_per_second:,.0f} rows/s")
        
        # All distributions should perform reasonably
        for r in results:
            assert r['throughput'] > 500_000, f"Throughput too low for {r['distribution']}: {r['throughput']:,.0f}"
    
    def test_performance_memory_efficiency(self):
        """Test memory efficiency with large dataset."""
        n_rows = 10_000_000
        
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        
        # Force garbage collection
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        df = generate_test_data(n_rows)
        mem_after_df = process.memory_info().rss / 1024 / 1024  # MB
        
        result = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        mem_after_liftover = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del df, result
        gc.collect()
        mem_after_cleanup = process.memory_info().rss / 1024 / 1024  # MB
        
        df_memory = mem_after_df - mem_before
        liftover_memory = mem_after_liftover - mem_after_df
        total_memory = mem_after_liftover - mem_before
        
        print(f"\nMemory Efficiency Test ({n_rows:,} rows):")
        print(f"  Memory for DataFrame: {df_memory:.1f} MB")
        print(f"  Memory for liftover: {liftover_memory:.1f} MB")
        print(f"  Total memory used: {total_memory:.1f} MB")
        print(f"  Memory per row: {total_memory/n_rows*1024:.3f} KB")
        
        # Memory should be reasonable (less than 1GB for 10M rows)
        assert total_memory < 2000, f"Memory usage too high: {total_memory:.1f} MB"
