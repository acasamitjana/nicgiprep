"""
Pipelines for NiGiPrep preprocessing workflows.

This subpackage contains the base and specialized pipelines for
cross-sectional, longitudinal, and multimodal data processing.
"""

from .base import Processor
from .cross_sectional import CrossSectionalProcessor, T1wSegmentationProcessor, T1wBiasCorrectionProcessor, MNIRegistrationProcessor
from .longitudinal import LongitudinalProcessor
from .multimodal import MMProcessor, MultiModalSegmentationProcessor, MultiModalBiasCorrectionProcessor, MultiMRIProcessor

__all__ = [
    "Processor",
    "CrossSectionalProcessor",
    "T1wSegmentationProcessor",
    "T1wBiasCorrectionProcessor",
    "MNIRegistrationProcessor",
    "LongitudinalProcessor",
    "MMProcessor",
    "MultiModalSegmentationProcessor",
    "MultiModalBiasCorrectionProcessor",
    "MultiMRIProcessor"
]