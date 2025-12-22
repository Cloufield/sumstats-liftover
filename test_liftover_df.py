"""
Tests for liftover_df module.
"""

import pandas as pd
import pytest
from sumstats_liftover import liftover_df


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
        chain_path="hg19ToHg38.over.chain.gz",
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
    
    # Check that chromosome is still 1
    assert (result['CHR_LIFT'] == 1).all()
    
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
        chain_path="hg19ToHg38.over.chain.gz",
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

