"""Tests for HSP calibration reference table (interpretive, not pipeline targets)."""

from __future__ import annotations

import pytest

from rde_domains.hsp_functions.complexity import calibration_reference


def test_calibration_reference_known_families():
    simon = calibration_reference("simon", 16)
    assert simon.family == "simon"
    assert simon.q_quantum_queries < simon.q_classical_queries
    assert simon.log2_gap > 0

    shor = calibration_reference("shor_cyclic", 16)
    assert shor.quantum_basis.startswith("cyclic")

    dihedral = calibration_reference("dihedral_kuperberg", 16)
    assert dihedral.q_quantum_queries >= 2.0


def test_calibration_reference_unknown_family():
    with pytest.raises(ValueError, match="no closed-form"):
        calibration_reference("unknown_family", 4)
