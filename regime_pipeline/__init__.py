"""Unified two-stage regime pipeline."""

from .regime_detection.pipeline import RegimeDetectionResult, run_regime_detection

__all__ = ["RegimeDetectionResult", "run_regime_detection"]
