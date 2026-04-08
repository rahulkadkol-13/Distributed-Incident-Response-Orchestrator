"""Deterministic incident response environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from models import Action, Observation, ObservationMetrics


ALLOWED_ACTIONS = {
    "restart_service",
    "scale_resources",
    "alert_engineer",
    "reroute_traffic",
    "ignore",
}

INCIDENT_ACTION_MAP: dict[str, str] = {
    "server_down": "restart_service",
    "api_crash": "restart_service",
    "overload": "scale_resources",
    "api_latency": "reroute_traffic",
    "network_outage": "reroute_traffic",
    "database_alert": "alert_engineer",
}

INCIDENT_POOL = tuple(INCIDENT_ACTION_MAP.keys())


@dataclass(slots=True)
class IncidentRecord:
    """Internal incident representation."""

    incident_type: str
    severity: int
    created_step: int
    response_steps: int = 0


@dataclass(slots=True)
class EpisodeMetrics:
    """Cumulative metrics for the environment instance."""

    total_reward: float = 0.0
    incidents_spawned: int = 0
    incidents_resolved: int = 0
    total_response_time: int = 0
    episodes_played: int = 0
    episodes_successful: int = 0
    episodes_failed: int = 0


class IncidentEnvironment:
    """OpenEnv-compatible incident response simulator.

    The environment models a single operational incident queue where an agent
    must choose a response action under time pressure. The state is fully
    deterministic for a given seed and action sequence.
    """

    def __init__(
        self,
        seed: int | None = 42,
        episode_length: int = 12,
        max_active_incidents: int = 3,
    ) -> None:
        self.seed = seed
        self.episode_length = episode_length
        self.max_active_incidents = max_active_incidents
        self.rng = random.Random(seed)
        self.metrics = EpisodeMetrics()

        self.system_load = 0
        self.time_remaining = 0
        self.resources_available = 0
        self.current_step = 0
        self.active_incidents: list[IncidentRecord] = []
        self.terminated = False
        self.termination_reason: str | None = None
        self.recent_action: str | None = None
        self._current_episode_reward = 0.0

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the environment to a deterministic initial state."""

        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

        self.system_load = self.rng.randint(45, 65)
        self.resources_available = self.rng.randint(2, 4)
        self.time_remaining = self.episode_length
        self.current_step = 0
        self.active_incidents = [self._generate_incident(start_step=0)]
        self.terminated = False
        self.termination_reason = None
        self.recent_action = None
        self._current_episode_reward = 0.0

        return self.state()

    def step(self, action: Action | dict[str, Any] | str) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply an action and advance the simulation by one step."""

        if self.terminated:
            observation = self.state()
            return observation, 0.0, True, {"already_terminated": True, "termination_reason": self.termination_reason}

        action_name, valid_action = self._normalize_action(action)
        self.recent_action = action_name

        reward = -2.0
        info: dict[str, Any] = {"valid_action": valid_action, "action": action_name}
        current_incident = self.active_incidents[0] if self.active_incidents else None

        if not valid_action:
            reward -= 10.0
            info["invalid_action"] = True
            self.system_load = min(100, self.system_load + 3)
        elif current_incident is None:
            reward -= 10.0
            info["no_active_incident"] = True
        elif self._is_correct_action(action_name, current_incident.incident_type):
            reward += float(10 * current_incident.severity)
            resolved_incident = self.active_incidents.pop(0)
            self._register_resolution(resolved_incident)
            self._apply_resolution_effects(action_name, resolved_incident.severity)
            info["resolved_incident"] = resolved_incident.incident_type
        else:
            reward -= 10.0
            self.system_load = min(100, self.system_load + 4)
            if self.active_incidents:
                self.active_incidents[0].response_steps += 1
            info["wrong_action"] = True

        self.current_step += 1
        self.time_remaining = max(0, self.time_remaining - 1)

        for incident in self.active_incidents:
            incident.response_steps += 1
            self.system_load = min(100, self.system_load + 1)

        if self.system_load >= 100 or (self.resources_available <= 0 and self.active_incidents):
            self._finish_episode("system_failure")
            reward -= 25.0
            info["failure_reason"] = self.termination_reason
        elif self.time_remaining == 0:
            if self._is_stable():
                self._finish_episode("system_stabilized")
                reward += 20.0
            else:
                self._finish_episode("time_exhausted")
                reward -= 25.0
            info["termination_reason"] = self.termination_reason
        elif self._is_stable():
            self._finish_episode("system_stabilized")
            reward += 20.0
            info["termination_reason"] = self.termination_reason
        elif not self.terminated:
            self._maybe_spawn_incident()
            if self.system_load >= 100 or (self.resources_available <= 0 and self.active_incidents):
                self._finish_episode("system_failure")
                reward -= 25.0
                info["failure_reason"] = self.termination_reason

        self._current_episode_reward += reward
        self.metrics.total_reward += reward

        observation = self.state()
        done = self.terminated
        info["terminated"] = done
        info["observation"] = observation.model_dump()
        info["metrics"] = observation.metrics.model_dump()
        info["episode_reward"] = self._current_episode_reward
        return observation, reward, done, info

    def state(self) -> Observation:
        """Return the current observation without mutating the environment."""

        incident = self.active_incidents[0] if self.active_incidents else None
        metrics = self._build_metrics()
        return Observation(
            incident_type=incident.incident_type if incident else "none",
            severity=incident.severity if incident else 0,
            system_load=self.system_load,
            time_remaining=self.time_remaining,
            active_incidents=len(self.active_incidents),
            resources_available=self.resources_available,
            current_step=self.current_step,
            terminated=self.terminated,
            termination_reason=self.termination_reason,
            recent_action=self.recent_action,
            metrics=metrics,
        )

    def _normalize_action(self, action: Action | dict[str, Any] | str) -> tuple[str, bool]:
        if isinstance(action, Action):
            return action.action, True

        if isinstance(action, str):
            return action, action in ALLOWED_ACTIONS

        if isinstance(action, dict):
            try:
                parsed = Action.model_validate(action)
                return parsed.action, True
            except Exception:
                raw_action = str(action.get("action", ""))
                return raw_action, raw_action in ALLOWED_ACTIONS

        return str(action), False

    def _generate_incident(self, start_step: int) -> IncidentRecord:
        incident_type = self.rng.choice(INCIDENT_POOL)
        severity = self.rng.randint(1, 5)
        self.metrics.incidents_spawned += 1
        return IncidentRecord(incident_type=incident_type, severity=severity, created_step=start_step)

    def _maybe_spawn_incident(self) -> None:
        if len(self.active_incidents) >= self.max_active_incidents:
            return

        spawn_chance = 0.35 if self.system_load >= 60 else 0.2
        if self.rng.random() < spawn_chance:
            incident = self._generate_incident(self.current_step)
            self.active_incidents.append(incident)
            self.system_load = min(100, self.system_load + incident.severity * 3)

    def _is_correct_action(self, action_name: str, incident_type: str) -> bool:
        return INCIDENT_ACTION_MAP.get(incident_type) == action_name

    def _apply_resolution_effects(self, action_name: str, severity: int) -> None:
        if action_name == "restart_service":
            self.system_load = max(0, self.system_load - severity * 6)
            self.resources_available = max(0, self.resources_available - 1)
        elif action_name == "scale_resources":
            self.system_load = max(0, self.system_load - severity * 4)
            self.resources_available = min(6, self.resources_available + 1)
        elif action_name == "reroute_traffic":
            self.system_load = max(0, self.system_load - severity * 5)
            self.resources_available = max(0, self.resources_available - 1)
        elif action_name == "alert_engineer":
            self.system_load = max(0, self.system_load - severity * 3)

        self.system_load = max(0, self.system_load)

    def _register_resolution(self, incident: IncidentRecord) -> None:
        response_time = self.current_step - incident.created_step + 1
        self.metrics.incidents_resolved += 1
        self.metrics.total_response_time += response_time

    def _is_stable(self) -> bool:
        return not self.active_incidents and self.system_load <= 30

    def _finish_episode(self, reason: str) -> None:
        if self.terminated:
            return

        self.terminated = True
        self.termination_reason = reason
        self.metrics.episodes_played += 1
        if reason == "system_stabilized":
            self.metrics.episodes_successful += 1
        else:
            self.metrics.episodes_failed += 1

    def _build_metrics(self) -> ObservationMetrics:
        episodes_played = self.metrics.episodes_played
        success_rate = self.metrics.episodes_successful / episodes_played if episodes_played else 0.0
        failure_rate = self.metrics.episodes_failed / episodes_played if episodes_played else 0.0
        average_response_time = (
            self.metrics.total_response_time / self.metrics.incidents_resolved
            if self.metrics.incidents_resolved
            else 0.0
        )

        return ObservationMetrics(
            total_reward=self.metrics.total_reward,
            success_rate=success_rate,
            failure_rate=failure_rate,
            average_response_time=average_response_time,
            incidents_resolved=self.metrics.incidents_resolved,
            incidents_spawned=self.metrics.incidents_spawned,
            episodes_played=episodes_played,
            episodes_successful=self.metrics.episodes_successful,
            episodes_failed=self.metrics.episodes_failed,
        )
