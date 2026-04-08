"""OpenEnv server entrypoint."""

from server.environment import IncidentEnvironment


def main():
    """
    Required entrypoint for multi-mode deployment.
    Returns the environment instance.
    """
    return IncidentEnvironment(seed=42)


# Some runtimes import `app`
app = main()


if __name__ == "__main__":
    # Ensures callable entrypoint
    main()