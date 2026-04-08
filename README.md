# Distributed Incident Response Orchestrator

This project is a deterministic OpenEnv-style simulation where an agent manages infrastructure incidents under time pressure.

## Problem

The environment simulates a production incident queue with server failures, overloads, API crashes, and routing problems. The agent must pick the right operational response before time runs out or the system fails.

## State

The observation includes:

- `incident_type`
- `severity`
- `system_load`
- `time_remaining`
- `active_incidents`
- `resources_available`
- `current_step`
- `terminated`
- `termination_reason`
- `recent_action`
- `metrics`

## Actions

Supported actions:

- `restart_service`
- `scale_resources`
- `alert_engineer`
- `reroute_traffic`
- `ignore`

## Reward

The reward scheme is severity-weighted and deterministic:

- Correct resolution: `+10 × severity`
- Delay penalty: `-2` per step
- Wrong action: `-10`
- System failure: `-25`
- System stabilized: `+20`

## Metrics

The environment tracks:

- `total_reward`
- `success_rate`
- `failure_rate`
- `average_response_time`
- `incidents_resolved`

It also tracks cumulative episode counts and incident throughput.

## How To Run

Install dependencies and run the local demo:

```bash
pip install -r requirements.txt
python client.py
```

Run the lightweight web interface locally:

```bash
python client.py --serve --host 0.0.0.0 --port 7860
```

## Deployment

The repository includes a Dockerfile for Hugging Face Spaces.

Build and run locally:

```bash
docker build -t incident-response-env .
docker run --rm -p 7860:7860 incident-response-env
```

For Spaces, configure the Docker Space to use the repository root. The container starts the web demo automatically.

## OpenEnv Entry Point

The environment entry point is defined in `openenv.yaml` and resolves to the `IncidentEnvironment` class.
## Termination Conditions

An episode ends when:

- time runs out
- the system stabilizes
- the system fails

## Determinism

The environment supports deterministic execution via a seed parameter:

```python
env = IncidentEnvironment(seed=42)
'''
## Project Structure


incident-response-env/
│
├── server/
│ └── environment.py
├── client.py
├── models.py
├── Dockerfile
├── openenv.yaml
├── README.md
└── requirements.txt

## Invalid Action Handling

Invalid actions are safely handled and penalized without crashing the environment.