import os
from server.environment import IncidentEnvironment

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")


def run_episode():
    print("START")

    env = IncidentEnvironment(seed=42)
    state = env.reset()

    done = False
    step_count = 0

    while not done:
        print(f"STEP {step_count}: {state}")

        # Simple baseline policy
        action = "restart_service"

        state, reward, done, info = env.step(action)

        print(f"ACTION: {action}, REWARD: {reward}")

        step_count += 1

    print("END")
    print("FINAL STATE:", state)


if __name__ == "__main__":
    run_episode()