"""
Tests for built-in chain files functionality.
"""

import os
import pytest
from sumstats_liftover import get_chain_path, list_chain_files, get_chain_info

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_list_chain_files():
    """Test listing available chain files."""
    files = list_chain_files()
    
    assert isinstance(files, dict)
    assert len(files) > 0
    assert "hg19ToHg38" in files
    assert "hg38ToHg19" in files
    assert "hg18ToHg19" in files
    
    # Check that descriptions are present
    for name, desc in files.items():
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_get_chain_path():
    """Test getting path to built-in chain files."""
    # Test with short name
    path1 = get_chain_path("hg19ToHg38")
    assert os.path.exists(path1)
    assert path1.endswith("hg19ToHg38.over.chain.gz")
    
    # Test with full filename
    path2 = get_chain_path("hg19ToHg38.over.chain.gz")
    assert path1 == path2
    
    # Test other chain files
    path3 = get_chain_path("hg38ToHg19")
    assert os.path.exists(path3)
    
    path4 = get_chain_path("hg18ToHg19")
    assert os.path.exists(path4)


def test_get_chain_path_invalid():
    """Test error handling for invalid chain file names."""
    with pytest.raises(ValueError, match="Unknown chain file"):
        get_chain_path("invalid_chain_file")
    
    with pytest.raises(ValueError, match="Unknown chain file"):
        get_chain_path("nonexistent")


def test_get_chain_info():
    """Test getting information about chain files."""
    info = get_chain_info("hg19ToHg38")
    
    assert isinstance(info, dict)
    assert info["name"] == "hg19ToHg38"
    assert info["filename"] == "hg19ToHg38.over.chain.gz"
    assert "description" in info
    assert "path" in info
    assert "exists" in info
    assert "size" in info
    assert "size_mb" in info
    
    assert info["exists"] is True
    assert info["size"] > 0
    assert info["size_mb"] > 0
    
    # Test description
    assert "hg19" in info["description"].lower() or "grch37" in info["description"].lower()
    assert "hg38" in info["description"].lower() or "grch38" in info["description"].lower()


def test_get_chain_info_all():
    """Test getting info for all available chain files."""
    files = list_chain_files()
    
    for name in files.keys():
        info = get_chain_info(name)
        assert info["exists"] is True, f"Chain file {name} should exist"
        assert info["size"] > 0, f"Chain file {name} should have size > 0"


def test_use_built_in_chain_file():
    """Test using built-in chain file with liftover_df."""
    from sumstats_liftover import liftover_df
    import pandas as pd
    
    # Create test dataframe
    df = pd.DataFrame({
        'CHR': [1, 1],
        'POS': [725932, 725933],
    })
    
    # Use built-in chain file
    chain_path = get_chain_path("hg19ToHg38")
    result = liftover_df(
        df,
        chain_path=chain_path,
        chrom_col="CHR",
        pos_col="POS"
    )
    
    # Verify results
    assert len(result) == len(df)
    assert 'CHR_LIFT' in result.columns
    assert 'POS_LIFT' in result.columns
    assert result['POS_LIFT'].notna().all()

