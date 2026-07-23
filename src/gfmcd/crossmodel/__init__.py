"""Cross-model carrier transfer instruments (Phase 2).

Port of the SpliceBERT positional-carrier assays to other single-nt transformer MLMs
(RNA-FM, RNA-BERT). The two objects a carrier is defined on -- a learned absolute
position table P and attention heads -- exist in these models, so the SpliceBERT
instruments transfer with only the P-attribute path abstracted away.
"""
from gfmcd.crossmodel.carrier_spectrum import (
    get_position_table,
    leakage_aware_band_power,
    periodogram,
    carrier_test,
    exact_bin_band_ratio,
)

__all__ = [
    "get_position_table",
    "leakage_aware_band_power",
    "periodogram",
    "carrier_test",
    "exact_bin_band_ratio",
]
