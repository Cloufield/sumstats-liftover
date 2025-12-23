"""
Built-in chain files for common genome build conversions.

This module provides access to commonly used UCSC chain files that are
included with the package, eliminating the need to download them separately.
"""

import os
from pathlib import Path

# Get the package data directory
_PACKAGE_DIR = Path(__file__).parent
_DATA_DIR = _PACKAGE_DIR / "data"

# Available built-in chain files
AVAILABLE_CHAIN_FILES = {
    "hg19ToHg38": "hg19ToHg38.over.chain.gz",
    "hg38ToHg19": "hg38ToHg19.over.chain.gz",
    "hg18ToHg19": "hg18ToHg19.over.chain.gz",
}

# Descriptions for each chain file
CHAIN_FILE_DESCRIPTIONS = {
    "hg19ToHg38": "Convert from hg19/GRCh37 to hg38/GRCh38",
    "hg38ToHg19": "Convert from hg38/GRCh38 to hg19/GRCh37",
    "hg18ToHg19": "Convert from hg18 to hg19/GRCh37",
}


def get_chain_path(name: str) -> str:
    """
    Get the path to a built-in chain file.
    
    Parameters
    ----------
    name : str
        Name of the chain file. Can be:
        - "hg19ToHg38" or "hg19ToHg38.over.chain.gz"
        - "hg38ToHg19" or "hg38ToHg19.over.chain.gz"
        - "hg18ToHg19" or "hg18ToHg19.over.chain.gz"
    
    Returns
    -------
    str
        Absolute path to the chain file
    
    Raises
    ------
    FileNotFoundError
        If the chain file is not found
    ValueError
        If the chain file name is not recognized
    
    Examples
    --------
    >>> from sumstats_liftover import get_chain_path
    >>> chain_path = get_chain_path("hg19ToHg38")
    >>> result = liftover_df(df, chain_path=chain_path)
    """
    # Normalize the name (remove .over.chain.gz if present)
    name_clean = name.replace(".over.chain.gz", "")
    
    if name_clean not in AVAILABLE_CHAIN_FILES:
        available = ", ".join(AVAILABLE_CHAIN_FILES.keys())
        raise ValueError(
            f"Unknown chain file: '{name}'. "
            f"Available chain files: {available}"
        )
    
    filename = AVAILABLE_CHAIN_FILES[name_clean]
    chain_path = _DATA_DIR / filename
    
    if not chain_path.exists():
        raise FileNotFoundError(
            f"Chain file not found: {chain_path}. "
            f"This may indicate an installation issue."
        )
    
    return str(chain_path.absolute())


def list_chain_files() -> dict:
    """
    List all available built-in chain files.
    
    Returns
    -------
    dict
        Dictionary mapping chain file names to their descriptions
    
    Examples
    --------
    >>> from sumstats_liftover import list_chain_files
    >>> files = list_chain_files()
    >>> for name, desc in files.items():
    ...     print(f"{name}: {desc}")
    """
    return {
        name: CHAIN_FILE_DESCRIPTIONS.get(name, "No description available")
        for name in AVAILABLE_CHAIN_FILES.keys()
    }


def get_chain_info(name: str) -> dict:
    """
    Get information about a built-in chain file.
    
    Parameters
    ----------
    name : str
        Name of the chain file
    
    Returns
    -------
    dict
        Dictionary with chain file information:
        - name: Chain file name
        - path: Full path to the chain file
        - description: Description of what the chain file does
        - size: File size in bytes
        - exists: Whether the file exists
    
    Examples
    --------
    >>> from sumstats_liftover import get_chain_info
    >>> info = get_chain_info("hg19ToHg38")
    >>> print(f"File size: {info['size']} bytes")
    """
    name_clean = name.replace(".over.chain.gz", "")
    
    if name_clean not in AVAILABLE_CHAIN_FILES:
        available = ", ".join(AVAILABLE_CHAIN_FILES.keys())
        raise ValueError(
            f"Unknown chain file: '{name}'. "
            f"Available chain files: {available}"
        )
    
    filename = AVAILABLE_CHAIN_FILES[name_clean]
    chain_path = _DATA_DIR / filename
    
    info = {
        "name": name_clean,
        "filename": filename,
        "path": str(chain_path.absolute()),
        "description": CHAIN_FILE_DESCRIPTIONS.get(name_clean, "No description"),
        "exists": chain_path.exists(),
    }
    
    if chain_path.exists():
        info["size"] = chain_path.stat().st_size
        info["size_mb"] = round(info["size"] / (1024 * 1024), 2)
    else:
        info["size"] = None
        info["size_mb"] = None
    
    return info

