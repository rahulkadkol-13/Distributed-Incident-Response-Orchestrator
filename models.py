"""Pydantic models for the incident response environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AllowedAction = Literal[
    "restart_service",
    "scale_resources",
    "alert_engineer",
    "reroute_traffic",
    "ignore",
]


class Action(BaseModel):
    """Validated action submitted to the environment."""

    model_config = ConfigDict(extra="forbid")

    action: AllowedAction = Field(..., description="Incident response action.")


class ObservationMetrics(BaseModel):
    """Evaluation metrics exposed in the observation."""

    model_config = ConfigDict(extra="forbid")

    total_reward: float = Field(..., description="Cumulative reward across episodes.")
    success_rate: float = Field(..., ge=0.0, le=1.0)
    failure_rate: float = Field(..., ge=0.0, le=1.0)
    average_response_time: float = Field(..., ge=0.0)
    incidents_resolved: int = Field(..., ge=0)
    incidents_spawned: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    episodes_successful: int = Field(..., ge=0)
    episodes_failed: int = Field(..., ge=0)


class Observation(BaseModel):
    """Observation returned by reset, step, and state."""

    model_config = ConfigDict(extra="forbid")

    incident_type: str = Field(..., description="Current incident type or 'none'.")
    severity: int = Field(..., ge=0, le=5)
    system_load: int = Field(..., ge=0, le=100)
    time_remaining: int = Field(..., ge=0)
    active_incidents: int = Field(..., ge=0)
    resources_available: int = Field(..., ge=0)
    current_step: int = Field(..., ge=0)
    terminated: bool
    termination_reason: str | None = None
    recent_action: str | None = None
    metrics: ObservationMetrics
