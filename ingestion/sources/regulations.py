"""Registry of supported EU regulations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegulationMeta:
    """Metadata for a single EU regulation."""

    regulation_id: str
    celex: str
    full_name: str
    short_name: str
    topics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGULATIONS: dict[str, RegulationMeta] = {
    "DORA": RegulationMeta(
        regulation_id="DORA",
        celex="32022R2554",
        full_name="Regulation (EU) 2022/2554",
        short_name="DORA",
        topics=["ict_risk", "digital_resilience", "financial_sector", "incident_reporting"],
    ),
    "NIS2": RegulationMeta(
        regulation_id="NIS2",
        celex="32022L2555",
        full_name="Directive (EU) 2022/2555",
        short_name="NIS2",
        topics=[
            "cybersecurity",
            "critical_infrastructure",
            "incident_reporting",
            "supply_chain",
        ],
    ),
}


def get_regulation(regulation_id: str) -> RegulationMeta:
    """Return metadata for a regulation by ID (case-insensitive).

    Raises:
        KeyError: If the regulation is not found.
    """
    key = regulation_id.upper()
    if key not in REGULATIONS:
        raise KeyError(
            f"Unknown regulation '{regulation_id}'. "
            f"Available: {list(REGULATIONS.keys())}"
        )
    return REGULATIONS[key]


def list_regulations() -> list[str]:
    """Return all registered regulation IDs."""
    return list(REGULATIONS.keys())
