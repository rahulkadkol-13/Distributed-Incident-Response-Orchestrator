"""OpenEnv server entrypoint."""

from server.environment import IncidentEnvironment


def create_app():
    """Factory function required for OpenEnv multi-mode deployment."""
    return IncidentEnvironment(seed=42)


# Some runtimes import `app` directly
app = create_app()