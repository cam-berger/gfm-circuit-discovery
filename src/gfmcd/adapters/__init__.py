"""Adapters that give pretrained genomic models a common circuit-facing API."""

from gfmcd.adapters.splicebert import SpliceBertAdapter, SpliceBertMetadata

__all__ = ["SpliceBertAdapter", "SpliceBertMetadata"]
