"""
LSEG Test Case Generators

This package contains generators for creating comprehensive test cases
for the LSEG financial data evaluation platform.

Generators:
- lookup_generator: Single/multi-field lookups
- temporal_generator: Temporal variant expansion
- comparison_generator: Multi-ticker comparisons
- screening_generator: Screening query generation
- error_generator: Error scenario generation
- complex_generator: Multi-step workflow generation
- entity_resolution_generator: Entity name/ticker to RIC resolution tests
"""

from .lookup_generator import LookupGenerator
from .temporal_generator import TemporalGenerator
from .comparison_generator import ComparisonGenerator
from .screening_generator import ScreeningGenerator
from .error_generator import ErrorGenerator
from .complex_generator import ComplexGenerator
from .entity_resolution_generator import EntityResolutionGenerator

__all__ = [
    'LookupGenerator',
    'TemporalGenerator',
    'ComparisonGenerator',
    'ScreeningGenerator',
    'ErrorGenerator',
    'ComplexGenerator',
    'EntityResolutionGenerator',
]
