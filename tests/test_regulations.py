"""Tests for the regulation registry."""

import pytest
from ingestion.sources.regulations import (
    REGULATIONS,
    get_regulation,
    list_regulations,
)


def test_registry_has_dora():
    assert "DORA" in REGULATIONS


def test_registry_has_nis2():
    assert "NIS2" in REGULATIONS


def test_dora_metadata():
    dora = REGULATIONS["DORA"]
    assert dora.celex == "32022R2554"
    assert "Regulation" in dora.full_name
    assert "2022/2554" in dora.full_name
    assert "ict_risk" in dora.topics
    assert "digital_resilience" in dora.topics
    assert "financial_sector" in dora.topics
    assert "incident_reporting" in dora.topics


def test_nis2_metadata():
    nis2 = REGULATIONS["NIS2"]
    assert nis2.celex == "32022L2555"
    assert "Directive" in nis2.full_name
    assert "2022/2555" in nis2.full_name
    assert "cybersecurity" in nis2.topics
    assert "critical_infrastructure" in nis2.topics
    assert "incident_reporting" in nis2.topics
    assert "supply_chain" in nis2.topics


def test_get_regulation_case_insensitive():
    assert get_regulation("dora").regulation_id == "DORA"
    assert get_regulation("DORA").regulation_id == "DORA"
    assert get_regulation("Nis2").regulation_id == "NIS2"


def test_get_regulation_unknown_raises():
    with pytest.raises(KeyError, match="Unknown regulation"):
        get_regulation("GDPR")


def test_list_regulations():
    regs = list_regulations()
    assert "DORA" in regs
    assert "NIS2" in regs
    assert len(regs) >= 2


def test_regulation_immutable():
    """RegulationMeta is frozen — mutation should raise."""
    dora = get_regulation("DORA")
    with pytest.raises((AttributeError, TypeError)):
        dora.celex = "changed"  # type: ignore[misc]
