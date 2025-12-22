"""
Example usage of liftover_df with hg19ToHg38.over.chain.gz

This example demonstrates lifting over genomic coordinates from hg19 to hg38
using the provided dataframe format.
"""

import pandas as pd
from sumstats_liftover import liftover_df

# Example dataframe with hg19 positions
df = pd.DataFrame({
    'SNPID': ['1:725932_G_A', '1:725933_A_G', '1:737801_T_C'],
    'CHR': [1, 1, 1],
    'POS': [725932, 725933, 737801],  # hg19 positions
    'EA': ['G', 'A', 'C'],
    'NEA': ['A', 'G', 'T'],
    'STATUS': [3860099, 3860099, 3860099],
    'EAF': [0.9960, 0.0040, 0.0051],
    'BETA': [-0.0737, 0.0737, 0.0490],
    'SE': [0.1394, 0.1394, 0.1231],
    'P': [0.5970, 0.5973, 0.6908],
    'DIRECTION': ['-?+-', '+?-+', '+?-+'],
    'N': [166718, 166718, 166718]
})

print("Original dataframe (hg19 coordinates):")
print(df)
print("\n" + "="*80 + "\n")

# Perform liftover from hg19 to hg38
result = liftover_df(
    df,
    chain_path="hg19ToHg38.over.chain.gz",
    chrom_col="CHR",
    pos_col="POS",
    out_chrom_col="CHR_LIFT",
    out_pos_col="POS_LIFT",
    out_strand_col="STRAND_LIFT"
)

print("Lifted dataframe (hg38 coordinates):")
print(result)
print("\n" + "="*80 + "\n")

# Show mapping summary
print("Mapping summary:")
print(f"Total variants: {len(result)}")
mapped = result['POS_LIFT'].notna() & (result['POS_LIFT'] > 0)
print(f"Mapped variants: {mapped.sum()}")
print(f"Unmapped variants: {(~mapped).sum()}")
print("\n" + "="*80 + "\n")

# Show coordinate comparison
print("Coordinate comparison:")
comparison = result[['SNPID', 'CHR', 'POS', 'CHR_LIFT', 'POS_LIFT', 'STRAND_LIFT']].copy()
comparison.columns = ['SNPID', 'CHR_hg19', 'POS_hg19', 'CHR_hg38', 'POS_hg38', 'STRAND']
print(comparison)

