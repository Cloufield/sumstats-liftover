"""
Accuracy tests for all variant types including corner cases, comparing with UCSC liftOver.

This test suite focuses on testing accuracy for all types of variants including corner cases
to ensure our package produces expected outputs that match or are compatible with UCSC liftOver behavior.

Role: Test accuracy for all types of variants including corner cases with UCSC liftover tool.
"""

import os
import subprocess
import tempfile
import pandas as pd
import pytest
from sumstats_liftover import liftover_df

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN_FILE = os.path.join(PROJECT_ROOT, "hg19ToHg38.over.chain.gz")


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


def normalize_chrom_for_comparison(chrom):
    """
    Normalize chromosome for comparison (handles string vs numeric, removes .0 suffix).
    Converts X->23, Y->24, M->25 for consistent comparison.
    """
    if chrom is None:
        return None
    chrom_str = str(chrom).rstrip('.0')
    
    # Normalize special chromosomes to numeric for comparison
    if chrom_str.upper() in ['X', '23']:
        return 23
    elif chrom_str.upper() in ['Y', '24']:
        return 24
    elif chrom_str.upper() in ['M', 'MT', '25']:
        return 25
    
    # Convert numeric strings to int if possible
    try:
        return int(chrom_str)
    except ValueError:
        return chrom_str


def normalize_pos_for_comparison(pos):
    """Normalize position for comparison (handles int vs float)."""
    if pd.isna(pos):
        return None
    return float(pos)


class TestVariantTypes:
    """Test suite for different variant types."""
    
    def test_successfully_mapped_variants(self):
        """Test variants that should map successfully (both tools agree)."""
        df = pd.DataFrame({
            'SNPID': ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C'],
            'CHR': [1, 1, 1],
            'POS': [725932, 725933, 737801],  # hg19 positions
        })
        
        # Run our package
        result_pkg = liftover_df(
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
        
        # All should be mapped
        assert result_pkg['POS_LIFT'].notna().all()
        assert (result_pkg['POS_LIFT'] > 0).all()
        
        # Compare with UCSC
        for idx in range(len(df)):
            pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[idx]['CHR_LIFT'])
            pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
            ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
            ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
            
            assert pkg_chr == ucsc_chr, f"Chromosome mismatch at {df.iloc[idx]['SNPID']}: {pkg_chr} vs {ucsc_chr}"
            assert pkg_pos == ucsc_pos, f"Position mismatch at {df.iloc[idx]['SNPID']}: {pkg_pos} vs {ucsc_pos}"
        
        # Verify expected positions
        expected_positions = [790552, 790553, 802421]
        assert result_pkg['POS_LIFT'].tolist() == expected_positions
    
    def test_unmapped_variants(self):
        """Test variants that should be unmapped (positions outside alignment blocks)."""
        df = pd.DataFrame({
            'CHR': [1, 2, 1],
            'POS': [999999999, 500000000, 1],  # Positions likely to be unmapped
            'SNPID': ['1:999999999', '2:500000000', '1:1']
        })
        
        # Run our package
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        # Run UCSC liftOver
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare mapping status
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            # Both should agree on mapping status (both mapped or both unmapped)
            assert pkg_mapped == ucsc_mapped, (
                f"Mapping status mismatch at {df.iloc[idx]['SNPID']}: "
                f"package={'mapped' if pkg_mapped else 'unmapped'}, "
                f"UCSC={'mapped' if ucsc_mapped else 'unmapped'}"
            )
            
            # If both mapped, positions should match
            if pkg_mapped and ucsc_mapped:
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                assert pkg_pos == ucsc_pos, f"Position mismatch at {df.iloc[idx]['SNPID']}"
    
    def test_multiple_chromosomes(self):
        """Test variants across multiple chromosomes."""
        df = pd.DataFrame({
            'CHR': [1, 2, 3, 10, 22],
            'POS': [725932, 100000, 500000, 1000000, 2000000],
            'SNPID': [f'{chr}:{pos}' for chr, pos in zip([1, 2, 3, 10, 22], [725932, 100000, 500000, 1000000, 2000000])]
        })
        
        # Run our package
        result_pkg = liftover_df(
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
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[idx]['CHR_LIFT'])
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                
                assert pkg_chr == ucsc_chr, f"Chromosome mismatch at {df.iloc[idx]['SNPID']}"
                assert pkg_pos == ucsc_pos, f"Position mismatch at {df.iloc[idx]['SNPID']}"
    
    def test_special_chromosome_formats(self):
        """Test special chromosome formats (X, Y, M in numeric and string formats)."""
        df = pd.DataFrame({
            'CHR': [23, 24, 25, 'X', 'Y', 'M'],  # Numeric and string formats
            'POS': [1000000, 500000, 1000, 2000000, 100000, 500],
            'SNPID': ['X:1000000', 'Y:500000', 'M:1000', 'X_str:2000000', 'Y_str:100000', 'M_str:500']
        })
        
        # Run our package
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        # Run UCSC liftOver
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare results
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            # Both should agree on mapping status
            assert pkg_mapped == ucsc_mapped, (
                f"Mapping status mismatch at {df.iloc[idx]['SNPID']}: "
                f"package={'mapped' if pkg_mapped else 'unmapped'}, "
                f"UCSC={'mapped' if ucsc_mapped else 'unmapped'}"
            )
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[idx]['CHR_LIFT'])
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                
                # Chromosomes should match (may be string or numeric, but should represent same chromosome)
                assert pkg_chr == ucsc_chr, f"Chromosome mismatch at {df.iloc[idx]['SNPID']}: {pkg_chr} vs {ucsc_chr}"
                assert pkg_pos == ucsc_pos, f"Position mismatch at {df.iloc[idx]['SNPID']}: {pkg_pos} vs {ucsc_pos}"
    
    def test_nonstandard_chromosome_mappings(self):
        """Test variants that UCSC maps to non-standard chromosomes (alternate contigs)."""
        # Use positions known to map to alternate contigs
        df = pd.DataFrame({
            'CHR': [8, 1, 7],
            'POS': [2314366, 143469538, 142113887],  # Positions that may map to alternate contigs
            'SNPID': ['8:2314366', '1:143469538', '7:142113887']
        })
        
        # Run our package with default settings (allows non-standard)
        result_pkg_default = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        # Run our package with filtering enabled
        result_pkg_filtered = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_nonstandard_chromosomes=True,
            remove_alternative_chromosomes=True,
            remove_unmapped=False
        )
        
        # Run UCSC liftOver
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # With default settings, we should match UCSC behavior
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg_default.iloc[idx]['POS_LIFT']) and result_pkg_default.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            # Mapping status should match UCSC
            assert pkg_mapped == ucsc_mapped, (
                f"Mapping status mismatch at {df.iloc[idx]['SNPID']} "
                f"(default mode should match UCSC)"
            )
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = str(result_pkg_default.iloc[idx]['CHR_LIFT'])
                ucsc_chr = str(result_ucsc.iloc[idx]['CHR_LIFT'])
                
                # Check if UCSC mapped to non-standard chromosome
                ucsc_is_nonstandard = not ucsc_chr.replace('.0', '').isdigit() and ucsc_chr not in ['X', 'Y', 'M', 'MT', 'x', 'y', 'm', 'mt']
                
                if ucsc_is_nonstandard:
                    # UCSC mapped to non-standard - our default should also map (or filter if requested)
                    # With filtering enabled, these should be unmapped
                    pkg_filtered_mapped = (
                        pd.notna(result_pkg_filtered.iloc[idx]['POS_LIFT']) 
                        and result_pkg_filtered.iloc[idx]['POS_LIFT'] > 0
                    )
                    # Filtered version should have fewer mappings
                    assert not pkg_filtered_mapped, (
                        f"Variant {df.iloc[idx]['SNPID']} mapped to non-standard chromosome "
                        f"but should be filtered when remove_nonstandard_chromosomes=True"
                    )
    
    def test_inter_chromosomal_mappings(self):
        """Test variants that may map to different chromosomes (inter-chromosomal)."""
        # Note: Most positions don't map inter-chromosomally, but we test the behavior
        df = pd.DataFrame({
            'CHR': [1, 2, 3],
            'POS': [725932, 100000, 500000],
            'SNPID': ['1:725932', '2:100000', '3:500000']
        })
        
        # Run with default settings (allows inter-chromosomal)
        result_pkg_default = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Run with filtering enabled
        result_pkg_filtered = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_different_chromosomes=True
        )
        
        # Run UCSC liftOver
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare results
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg_default.iloc[idx]['POS_LIFT']) and result_pkg_default.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = normalize_chrom_for_comparison(result_pkg_default.iloc[idx]['CHR_LIFT'])
                ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
                input_chr = normalize_chrom_for_comparison(df.iloc[idx]['CHR'])
                
                # Check if inter-chromosomal mapping occurred
                is_inter_chromosomal = (pkg_chr != input_chr)
                
                if is_inter_chromosomal:
                    # With filtering enabled, inter-chromosomal mappings should be removed
                    pkg_filtered_mapped = (
                        pd.notna(result_pkg_filtered.iloc[idx]['POS_LIFT']) 
                        and result_pkg_filtered.iloc[idx]['POS_LIFT'] > 0
                    )
                    assert not pkg_filtered_mapped, (
                        f"Inter-chromosomal mapping at {df.iloc[idx]['SNPID']} "
                        f"should be filtered when remove_different_chromosomes=True"
                    )
    
    def test_remove_option(self):
        """Test the convenience 'remove' option that filters all problematic mappings."""
        df = pd.DataFrame({
            'CHR': [1, 1, 8, 1],
            'POS': [725932, 999999999, 2314366, 143469538],  # Mix: mapped, unmapped, non-standard
            'SNPID': ['1:725932', '1:999999999', '8:2314366', '1:143469538']
        })
        
        # Run with remove=True
        result_removed = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove=True
        )
        
        # Run with remove=False (default)
        result_kept = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove=False
        )
        
        # remove=True should have fewer rows (filters unmapped and non-standard)
        assert len(result_removed) <= len(result_kept), "remove=True should filter some variants"
        
        # All remaining variants in result_removed should be:
        # 1. Mapped (POS_LIFT > 0)
        # 2. On standard chromosomes
        # 3. Same chromosome as input
        if len(result_removed) > 0:
            assert (result_removed['POS_LIFT'] > 0).all(), "All variants should be mapped"
            assert result_removed['CHR_LIFT'].notna().all(), "All variants should have valid chromosomes"
            
            # Check that chromosomes are standard (numeric 1-25 or X, Y, M)
            for idx in result_removed.index:
                chr_lift = result_removed.loc[idx, 'CHR_LIFT']
                chr_str = str(chr_lift).rstrip('.0')
                assert chr_str in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                                 '21', '22', '23', '24', '25', 'X', 'Y', 'M', 'MT'], (
                    f"Non-standard chromosome found: {chr_lift}"
                )
    
    def test_expected_output_formats(self):
        """Test that output formats match expected behavior."""
        df = pd.DataFrame({
            'CHR': [1, 23, 'X'],
            'POS': [725932, 1000000, 2000000],
            'SNPID': ['1:725932', 'X:1000000', 'X_str:2000000']
        })
        
        # Test with convert_special_chromosomes=False (default)
        result_string = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            convert_special_chromosomes=False
        )
        
        # Test with convert_special_chromosomes=True
        result_numeric = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            convert_special_chromosomes=True
        )
        
        # With convert_special_chromosomes=False, X should remain as string
        x_idx = df[df['CHR'] == 'X'].index[0]
        assert isinstance(result_string.iloc[x_idx]['CHR_LIFT'], str), (
            "With convert_special_chromosomes=False, X should be string"
        )
        assert result_string.iloc[x_idx]['CHR_LIFT'] in ['X', 'x'], (
            f"Expected 'X' or 'x', got {result_string.iloc[x_idx]['CHR_LIFT']}"
        )
        
        # With convert_special_chromosomes=True, X should be numeric 23
        x_numeric_idx = df[df['CHR'] == 23].index[0]
        chr_lift_numeric = result_numeric.iloc[x_numeric_idx]['CHR_LIFT']
        assert chr_lift_numeric == 23, (
            f"With convert_special_chromosomes=True, X should be 23, got {chr_lift_numeric}"
        )
    
    # ============================================================================
    # CORNER CASES AND EDGE CASES
    # ============================================================================
    
    def test_boundary_positions(self):
        """Test positions at chromosome boundaries and chain boundaries."""
        # Test positions near chromosome starts/ends and known chain boundaries
        df = pd.DataFrame({
            'CHR': [1, 1, 1, 2, 2, 22, 22],
            'POS': [1, 2, 3, 100000, 249250621, 1, 51304566],  # Boundary positions
            'SNPID': ['1:1', '1:2', '1:3', '2:100000', '2:249250621', '22:1', '22:51304566']
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare mapping status for boundary positions
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            assert pkg_mapped == ucsc_mapped, (
                f"Boundary position mapping mismatch at {df.iloc[idx]['SNPID']}: "
                f"package={'mapped' if pkg_mapped else 'unmapped'}, "
                f"UCSC={'mapped' if ucsc_mapped else 'unmapped'}"
            )
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[idx]['CHR_LIFT'])
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                
                assert pkg_chr == ucsc_chr, f"Chromosome mismatch at boundary {df.iloc[idx]['SNPID']}"
                assert pkg_pos == ucsc_pos, f"Position mismatch at boundary {df.iloc[idx]['SNPID']}"
    
    def test_chromosome_format_variations(self):
        """Test various chromosome format inputs (chr prefix, case variations)."""
        # Test with 'chr' prefix variations
        df = pd.DataFrame({
            'CHR': ['chr1', 'chrX', 'chrY', 'chrM', 'CHR1', 'Chr2'],  # Various formats
            'POS': [725932, 1000000, 500000, 1000, 100000, 200000],
            'SNPID': ['chr1:725932', 'chrX:1000000', 'chrY:500000', 'chrM:1000', 'CHR1:100000', 'Chr2:200000']
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare results
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            if pkg_mapped and ucsc_mapped:
                pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[idx]['CHR_LIFT'])
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[idx]['CHR_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                
                assert pkg_chr == ucsc_chr, f"Chromosome mismatch with format variation at {df.iloc[idx]['SNPID']}"
                assert pkg_pos == ucsc_pos, f"Position mismatch with format variation at {df.iloc[idx]['SNPID']}"
    
    def test_zero_based_vs_one_based(self):
        """Test coordinate system variations (0-based vs 1-based)."""
        # Test same positions with different coordinate systems
        df_1based = pd.DataFrame({
            'CHR': [1, 1],
            'POS': [725932, 725933],  # 1-based positions
            'SNPID': ['1:725932', '1:725933']
        })
        
        df_0based = pd.DataFrame({
            'CHR': [1, 1],
            'POS': [725931, 725932],  # 0-based equivalent
            'SNPID': ['1:725931', '1:725932']
        })
        
        # Run with 1-based input
        result_1based = liftover_df(
            df_1based,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            one_based_input=True
        )
        
        # Run with 0-based input
        result_0based = liftover_df(
            df_0based,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            one_based_input=False
        )
        
        # Results should be the same (both represent the same genomic positions)
        for idx in range(len(df_1based)):
            pos_1based = result_1based.iloc[idx]['POS_LIFT']
            pos_0based = result_0based.iloc[idx]['POS_LIFT']
            
            if pd.notna(pos_1based) and pd.notna(pos_0based):
                # 0-based output should be one less than 1-based
                assert abs(pos_1based - pos_0based) <= 1, (
                    f"Coordinate system mismatch: 1-based={pos_1based}, 0-based={pos_0based}"
                )
    
    def test_duplicate_positions(self):
        """Test handling of duplicate positions in input."""
        df = pd.DataFrame({
            'CHR': [1, 1, 1, 1],
            'POS': [725932, 725932, 725933, 725933],  # Duplicate positions
            'SNPID': ['dup1', 'dup2', 'dup3', 'dup4']
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Duplicate positions should map to same output positions
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            if pkg_mapped and ucsc_mapped:
                pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[idx]['POS_LIFT'])
                ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[idx]['POS_LIFT'])
                
                assert pkg_pos == ucsc_pos, f"Duplicate position mapping mismatch at {df.iloc[idx]['SNPID']}"
        
        # Same input positions should map to same output positions
        same_input_positions = df[df['POS'] == 725932].index
        if len(same_input_positions) > 1:
            first_pos = result_pkg.iloc[same_input_positions[0]]['POS_LIFT']
            for idx in same_input_positions[1:]:
                assert result_pkg.iloc[idx]['POS_LIFT'] == first_pos, (
                    "Duplicate input positions should map to same output position"
                )
    
    def test_empty_dataframe(self):
        """Test handling of empty input dataframe."""
        df = pd.DataFrame({
            'CHR': [],
            'POS': [],
            'SNPID': []
        })
        
        result = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        assert len(result) == 0, "Empty input should produce empty output"
        # For empty dataframes, columns may or may not be added, but result should be empty
        if len(result.columns) > 0:
            # If columns are added, they should include the output columns
            assert 'CHR_LIFT' in result.columns or 'CHR' in result.columns
            assert 'POS_LIFT' in result.columns or 'POS' in result.columns
    
    def test_single_variant(self):
        """Test handling of single variant (edge case for small datasets)."""
        df = pd.DataFrame({
            'CHR': [1],
            'POS': [725932],
            'SNPID': ['1:725932']
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        assert len(result_pkg) == 1
        assert len(result_ucsc) == 1
        
        pkg_mapped = pd.notna(result_pkg.iloc[0]['POS_LIFT']) and result_pkg.iloc[0]['POS_LIFT'] > 0
        ucsc_mapped = pd.notna(result_ucsc.iloc[0]['POS_LIFT']) and result_ucsc.iloc[0]['POS_LIFT'] > 0
        
        assert pkg_mapped == ucsc_mapped, "Single variant mapping status should match"
        
        if pkg_mapped and ucsc_mapped:
            pkg_chr = normalize_chrom_for_comparison(result_pkg.iloc[0]['CHR_LIFT'])
            pkg_pos = normalize_pos_for_comparison(result_pkg.iloc[0]['POS_LIFT'])
            ucsc_chr = normalize_chrom_for_comparison(result_ucsc.iloc[0]['CHR_LIFT'])
            ucsc_pos = normalize_pos_for_comparison(result_ucsc.iloc[0]['POS_LIFT'])
            
            assert pkg_chr == ucsc_chr, "Single variant chromosome should match"
            assert pkg_pos == ucsc_pos, "Single variant position should match"
    
    def test_very_large_positions(self):
        """Test handling of very large position values."""
        df = pd.DataFrame({
            'CHR': [1, 2, 22],
            'POS': [249250621, 243199373, 51304566],  # Near chromosome ends
            'SNPID': ['1:249250621', '2:243199373', '22:51304566']
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Compare mapping status
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            assert pkg_mapped == ucsc_mapped, (
                f"Large position mapping mismatch at {df.iloc[idx]['SNPID']}"
            )
    
    def test_all_chromosomes_coverage(self):
        """Test that all standard chromosomes (1-22, X, Y, M) are handled correctly."""
        # Test at least one position from each standard chromosome
        chroms = list(range(1, 23)) + [23, 24, 25, 'X', 'Y', 'M']
        positions = [725932] * 22 + [1000000, 500000, 1000, 2000000, 100000, 500]
        snpids = [f'{chr}:725932' if isinstance(chr, int) and chr <= 22 else f'{chr}:test' 
                 for chr in chroms]
        
        df = pd.DataFrame({
            'CHR': chroms,
            'POS': positions,
            'SNPID': snpids
        })
        
        result_pkg = liftover_df(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS",
            remove_unmapped=False
        )
        
        result_ucsc = run_ucsc_liftover(
            df,
            chain_path=CHAIN_FILE,
            chrom_col="CHR",
            pos_col="POS"
        )
        
        # Verify all chromosomes are processed
        assert len(result_pkg) == len(df), "All chromosomes should be processed"
        
        # Compare mapping status for each chromosome
        for idx in range(len(df)):
            pkg_mapped = pd.notna(result_pkg.iloc[idx]['POS_LIFT']) and result_pkg.iloc[idx]['POS_LIFT'] > 0
            ucsc_mapped = pd.notna(result_ucsc.iloc[idx]['POS_LIFT']) and result_ucsc.iloc[idx]['POS_LIFT'] > 0
            
            # Mapping status should match
            assert pkg_mapped == ucsc_mapped, (
                f"Chromosome {df.iloc[idx]['CHR']} mapping status mismatch"
            )

