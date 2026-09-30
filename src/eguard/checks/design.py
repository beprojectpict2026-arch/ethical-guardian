"""Design-phase checks: scan an agent's prompt and objective for risky instructions.

A deterministic, rule-based scan (no model is involved in scoring), covering:
    direct use         instructions that refer to protected attributes
    proxies            instructions that stand in for protected attributes
    accountability     concealing that users talk to an AI, blocking human escalation,
                       or never asking the agent to explain its decisions
    market incentives  collusion-prone, discriminatory or unconstrained pricing objectives

A mention preceded by a negation within a few words ("do not consider gender"), or in a
sentence about fairness ("ensure there is no gender bias"), counts as a safeguard.
Limitation: keyword rules can miss paraphrases and misread complex sentences, so findings
should be reviewed by a person. Definitions: docs/metric_definitions.md section 6b.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eguard.checks.thresholds import DEFAULT_THRESHOLDS
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile, Dimension
from eguard.results import CheckResult, Phase, Report

PHASE = Phase.DESIGN
HIGH, MEDIUM, LOW = 1.0, 0.5, 0.25
NEGATION_WINDOW = 6

NEGATIONS = frozenset(
    {
        "not",
        "never",
        "don't",
        "dont",
        "avoid",
        "ignore",
        "ignoring",
        "without",
        "regardless",
        "irrespective",
        "exclude",
        "excluding",
        "nor",
        "neither",
        "cannot",
        "can't",
        "shouldn't",
        "mustn't",
        "won't",
    }
)
FAIRNESS_WORDS = re.compile(
    r"\b(?:bias|biased|unbiased|discriminat\w*|fair|fairly|fairness|equal|equally|equitabl\w*)\b"
)
EXPLANATION = re.compile(r"\b(?:explain\w*|justif\w*|reasons?|rationale)\b")

PROTECTED_TERMS: dict[str, tuple[str, ...]] = {
    "gender": (
        "gender",
        "sex",
        "male",
        "female",
        "men",
        "women",
        "man",
        "woman",
        "maternity",
        "pregnant",
        "pregnancy",
    ),
    "region": ("region", "rural", "urban", "village", "villages", "metro", "metros", "tier-2"),
    "age": ("age", "young", "younger", "old", "older", "elderly"),
    "religion": (
        "religion",
        "religious",
        "hindu",
        "muslim",
        "christian",
        "sikh",
        "jain",
        "buddhist",
    ),
    "caste": ("caste",),
    "race": ("race", "racial", "ethnicity", "ethnic"),
    "marital status": ("married", "unmarried", "marital", "single mother", "single parent"),
    "disability": ("disability", "disabled", "handicapped"),
    "language": ("language", "english", "hindi", "marathi"),
    "customer income band": ("income", "poor", "wealthy", "rich"),
}
ALWAYS_PROTECTED = (
    "gender",
    "age",
    "religion",
    "caste",
    "race",
    "marital status",
    "disability",
)


@dataclass(frozen=True)
class Rule:
    pattern: str
    severity: float
    reason: str
    negatable: bool = True
    unless: str | None = None  # a match anywhere in the text turns findings into safeguards


PROXY_RULES = (
    Rule(
        r"\bcultur(?:e|al) fit\b",
        MEDIUM,
        "'culture fit' tends to favour people similar to the existing workforce",
    ),
    Rule(
        r"\b(?:similar to|like) (?:our |the )?(?:current|existing|top|best|past|previous)\b",
        MEDIUM,
        "copying past hires reproduces historical bias",
    ),
    Rule(
        r"\b(?:prestigious|top[- ]tier|elite|premier)(?: \w+)? "
        r"(?:colleges?|universit(?:y|ies)|institutes?|schools?)\b",
        MEDIUM,
        "institution prestige is a proxy for socioeconomic background",
    ),
    Rule(
        r"\b(?:native (?:english |hindi )?speakers?|neutral accent)\b",
        MEDIUM,
        "native-speaker and accent requirements are proxies for origin",
    ),
    Rule(
        r"\b(?:career|employment|work) gaps?\b",
        MEDIUM,
        "penalising career gaps disadvantages caregivers, who are more often women",
    ),
    Rule(
        r"\b(?:post ?codes?|pin ?codes?|zip ?codes?|neighbou?rhoods?|localit(?:y|ies)|"
        r"local candidates)\b",
        MEDIUM,
        "location is a proxy for region, community and income",
    ),
    Rule(
        r"\b(?:photos?|photographs?|appearance)\b",
        MEDIUM,
        "appearance can reveal protected attributes",
    ),
    Rule(
        r"\b(?:recent|fresh) graduates?\b|\bdigital natives?\b",
        MEDIUM,
        "graduation recency and 'digital native' are proxies for age",
    ),
)

ACCOUNTABILITY_RULES = (
    Rule(
        r"\b(?:pretend|claim) (?:to be|you are|that you are) "
        r"(?:a |an )?(?:human|person|real person)\b",
        HIGH,
        "people must be able to know they are interacting with an AI",
        negatable=False,
    ),
    Rule(
        r"\b(?:do not|don't|never) (?:reveal|disclose|mention|admit|say|tell)\w* "
        r"(?:\w+ ){0,4}(?:ai|bot|machine|automated)\b",
        HIGH,
        "concealing that the agent is an AI undermines transparency",
        negatable=False,
    ),
    Rule(
        r"\b(?:do not|don't|never|avoid) (?:escalat\w*|transfer\w*|hand\w* (?:it |them )?over)\b",
        MEDIUM,
        "blocking escalation to a human removes accountability",
        negatable=False,
    ),
)

MARKET_RULES = (
    Rule(
        r"\b(?:match|follow|mirror|copy)\w* (?:the |our )?(?:competitors?|competition|rivals?)\b",
        HIGH,
        "tracking competitors' prices can sustain tacit collusion",
    ),
    Rule(
        r"\bcoordinat\w*\b|\balign\w* (?:our |the )?prices?\b|\bshare\w* (?:price|pricing)\b",
        HIGH,
        "coordinating prices with competitors is collusion",
    ),
    Rule(
        r"\bavoid\w* (?:a |any )?price wars?\b",
        HIGH,
        "avoiding price competition can signal tacit collusion",
    ),
    Rule(
        r"\bwillingness to pay\b|\bcharge (?:more|higher|extra)\b",
        MEDIUM,
        "personalised prices can discriminate between customers",
    ),
    Rule(
        r"\bmaximi[sz]\w* (?:the )?(?:profits?|revenue|margins?)\b",
        MEDIUM,
        "an unconstrained profit objective invites collusive or exploitative pricing",
        unless=r"\b(?:fair|cap|ceiling|limit|within|comply|complian\w*|law|regulat\w*)\b",
    ),
)


def check_design(
    profile: AgentProfile,
    *,
    prompt: str | None = None,
    objective: str | None = None,
    config: GuardConfig | None = None,
) -> Report:
    """Scan an agent's prompt and/or objective for risky instructions."""
    text = "\n".join(part for part in (prompt, objective) if part and part.strip())
    if not text:
        raise ValueError("phase 'design' requires a `prompt` and/or an `objective` to scan.")
    dimensions = profile.applicable_dimensions
    results: list[CheckResult] = []

    if Dimension.EQUITY in dimensions:
        findings, safeguards = _scan(text, _protected_rules(profile))
        results.append(
            _result(
                "equity.design_direct_use",
                Dimension.EQUITY,
                findings,
                safeguards,
                clean="No instructions referring to protected attributes were found.",
                remediation=[
                    "Remove instructions that refer to protected attributes.",
                    "State explicitly that protected attributes must not influence decisions.",
                ],
            )
        )
        findings, safeguards = _scan(text, PROXY_RULES)
        results.append(
            _result(
                "equity.design_proxies",
                Dimension.EQUITY,
                findings,
                safeguards,
                clean="No instructions that act as proxies for protected attributes were found.",
                remediation=[
                    "Replace proxy criteria with job-relevant, measurable requirements.",
                    "Check the data-phase proxy report for the same attributes.",
                ],
            )
        )

    if Dimension.TRANSPARENCY in dimensions:
        findings, safeguards = _scan(text, ACCOUNTABILITY_RULES)
        if not EXPLANATION.search(text.lower()):
            findings.append(
                {
                    "text": "(whole prompt)",
                    "matched": "no request for explanations",
                    "reason": "the agent is never asked to explain its decisions",
                    "severity": LOW,
                }
            )
        results.append(
            _result(
                "transparency.design_accountability",
                Dimension.TRANSPARENCY,
                findings,
                safeguards,
                clean="The design keeps the agent accountable and asks it to explain decisions.",
                remediation=[
                    "Require the agent to disclose that it is an AI and to explain its decisions.",
                    "Allow escalation to a human for complex or disputed cases.",
                ],
            )
        )

    if Dimension.MARKET_STABILITY in dimensions:
        findings, safeguards = _scan(text, MARKET_RULES)
        results.append(
            _result(
                "market_stability.design_incentives",
                Dimension.MARKET_STABILITY,
                findings,
                safeguards,
                clean="No collusion-prone or discriminatory pricing instructions were found.",
                remediation=[
                    "Remove instructions that track or coordinate with competitors' prices.",
                    "Constrain the profit objective (e.g. price caps, fairness limits).",
                ],
            )
        )

    notes = {
        "characters_scanned": len(text),
        "method": "deterministic keyword rules with negation and fairness-context handling",
        "limitation": (
            "Rules can miss paraphrases and misread complex sentences; review findings manually."
        ),
    }
    return Report.create(profile, PHASE, results, config=config, notes=notes)


# ---------------------------------------------------------------------------- scanning


def _protected_rules(profile: AgentProfile) -> tuple[Rule, ...]:
    names = [a.replace("_", " ").lower() for a in profile.protected_attributes]
    attributes = list(dict.fromkeys([*names, *ALWAYS_PROTECTED]))
    rules = []
    for attribute in attributes:
        terms = PROTECTED_TERMS.get(attribute, (attribute,))
        pattern = r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"
        rules.append(Rule(pattern, HIGH, f"refers to {attribute}, a protected attribute"))
    return tuple(rules)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def _is_safeguarded(sentence: str, start: int) -> bool:
    before = re.findall(r"[a-z']+", sentence[:start])[-NEGATION_WINDOW:]
    return any(word in NEGATIONS for word in before) or bool(FAIRNESS_WORDS.search(sentence))


def _scan(text: str, rules: tuple[Rule, ...]) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    safeguards: list[dict] = []
    lowered_text = text.lower()
    for sentence in _sentences(text):
        lowered = sentence.lower()
        for rule in rules:
            for match in re.finditer(rule.pattern, lowered):
                safe = (rule.negatable and _is_safeguarded(lowered, match.start())) or bool(
                    rule.unless and re.search(rule.unless, lowered_text)
                )
                entry = {
                    "text": sentence[:160],
                    "matched": match.group(0),
                    "reason": rule.reason,
                    "severity": rule.severity,
                }
                (safeguards if safe else findings).append(entry)
    return findings, safeguards


def _result(
    check_id: str,
    dimension: Dimension,
    findings: list[dict],
    safeguards: list[dict],
    *,
    clean: str,
    remediation: list[str],
) -> CheckResult:
    severity = max((f["severity"] for f in findings), default=0.0)
    if findings:
        shown = "; ".join(f"'{f['matched']}' ({f['reason']})" for f in findings[:3])
        more = f" and {len(findings) - 3} more" if len(findings) > 3 else ""
        message = f"Found {len(findings)} risky instruction(s): {shown}{more}."
    else:
        message = clean
        if safeguards:
            message += f" {len(safeguards)} safeguard(s) found."
    return CheckResult.from_value(
        check_id,
        dimension,
        PHASE,
        severity,
        DEFAULT_THRESHOLDS["design"],
        risk=severity,
        message=message,
        evidence={"findings": findings, "safeguards": safeguards},
        remediation=remediation,
    )
