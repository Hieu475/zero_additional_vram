"""Competitive baselines module for speculative decoding."""

from zassd.baselines.knapspec import KnapSpecController
from zassd.baselines.specbound import SpecBoundController

__all__ = ["KnapSpecController", "SpecBoundController"]
