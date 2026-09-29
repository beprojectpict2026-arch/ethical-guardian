"""Hiring scenario: synthetic candidate pools with controllable historical bias."""

from eguard.scenarios.hiring.generator import (
    EDUCATION_LEVELS,
    FEATURE_COLUMNS,
    HIDDEN_COLUMNS,
    LABEL_COLUMN,
    PROTECTED_COLUMNS,
    CandidatePoolConfig,
    applicant_view,
    generate_candidates,
    historical_records,
)

__all__ = [
    "EDUCATION_LEVELS",
    "FEATURE_COLUMNS",
    "HIDDEN_COLUMNS",
    "LABEL_COLUMN",
    "PROTECTED_COLUMNS",
    "CandidatePoolConfig",
    "applicant_view",
    "generate_candidates",
    "historical_records",
]
