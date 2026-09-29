"""Human-readable reports: render a Report as a self-contained HTML page.

The page has no external dependencies (styles are inline), so it can be opened offline,
attached to an email or archived with a build. All text is HTML-escaped.
"""

from __future__ import annotations

import json
import math
from typing import Any

from jinja2 import Environment, PackageLoader

from eguard.results import Report

__all__ = ["render_html"]

_ENV = Environment(
    loader=PackageLoader("eguard.reporting", "templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _number(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


_ENV.filters["number"] = _number
_ENV.filters["pretty"] = _pretty


def _tier_name(risk: float, boundaries: tuple[float, float, float]) -> str:
    """Tier name for colouring a risk bar, using the report's own tier boundaries."""
    low, moderate, high = boundaries
    if risk < low:
        return "low"
    if risk < moderate:
        return "moderate"
    if risk < high:
        return "high"
    return "critical"


def render_html(report: Report) -> str:
    """Render a report as a complete, self-contained HTML document."""
    import eguard  # imported here to avoid a circular import at module load

    template = _ENV.get_template("report.html.j2")
    boundaries = report.config.tier_boundaries
    return template.render(
        report=report,
        composite=report.composite_risk,
        tier=report.risk_tier,
        counts=report.status_counts,
        dimension_risks=report.dimension_risks,
        tier_of=lambda risk: _tier_name(risk, boundaries),
        eguard_version=eguard.__version__,
    )
