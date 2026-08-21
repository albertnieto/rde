"""Deprecated import path — use ``rde.discovery.state_dynamics`` instead."""

from __future__ import annotations

import warnings

from rde.discovery.state_dynamics import (
    DynamicsStepResult,
    StateDynamicsResult,
    predict_dynamics_step,
    predict_state_dynamics,
)

__all__ = [
    "DynamicsStepResult",
    "StateDynamicsResult",
    "predict_dynamics_step",
    "predict_state_dynamics",
]

warnings.warn(
    "rde.discovery.dynamics is deprecated; import from rde.discovery.state_dynamics",
    DeprecationWarning,
    stacklevel=2,
)
