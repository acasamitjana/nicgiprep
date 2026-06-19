"""
Pipelines for NiGiPrep preprocessing workflows.

This subpackage contains the base and specialized pipelines for
cross-sectional, longitudinal, and multimodal data processing.
"""

from .base import Processor
from .cross_sectional import CrossSectionalProcessor, SynthSegProcessor, BiasCorrectionProcessor, MNIRegistrationProcessor
from .longitudinal import LongitudinalProcessor
from .multimodal import MMProcessor, MultiModalSynthSegProcessor, MultiModalBiasCorrectionProcessor, MultiMRIProcessor

__all__ = [
    "Processor",
    "CrossSectionalProcessor",
    "SynthSegProcessor",
    "BiasCorrectionProcessor",
    "LongitudinalProcessor",
    "MMProcessor",
    "MultiModalSynthSegProcessor",
    "MultiModalBiasCorrectionProcessor",
    "MultiMRIProcessor"
]