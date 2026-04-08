"""Command-line demo and lightweight web server for the environment."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from server.environment import IncidentEnvironment


ACTION_BUTTONS = [
    "restart_service",
    "scale_resources",
    "alert_engineer",
    "reroute_traffic",
    "ignore",
]


def run_demo(seed: int | None = 42) -> None:
    """Execute a short deterministic rollout and print the results."""

    env = IncidentEnvironment(seed=seed)
    print(json.dumps(env.state().model_dump(), indent=2))

    action_sequence = [
        "restart_service",
        "scale_resources",
        "alert_engineer",
        "reroute_traffic",
        "ignore",
    ]

    for action_name in action_sequence:
        observation, reward, done, info = env.step(action_name)
        print(json.dumps({"action": action_name, "reward": reward, "done": done, "state": observation.model_dump(), "info": info}, indent=2))
        if done:
            break


def run_server(host: str, port: int, seed: int | None) -> None:
    """Run a minimal HTTP interface for Hugging Face Spaces deployment."""

    env = IncidentEnvironment(seed=seed)

    class RequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str, status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _render_page(self) -> str:
            state = env.state().model_dump()
            buttons = "".join(
                f'<a class="button" href="/step?action={action}">{action}</a>' for action in ACTION_BUTTONS
            )
            return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Distributed Incident Response Orchestrator</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; margin: 0; padding: 32px; }}
    .card {{ max-width: 920px; margin: 0 auto; background: rgba(15, 23, 42, 0.88); border: 1px solid #334155; border-radius: 18px; padding: 24px; box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35); }}
    h1 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .stat {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 14px; }}
    .buttons {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
    .button {{ display: inline-block; background: #38bdf8; color: #082f49; text-decoration: none; padding: 10px 14px; border-radius: 999px; font-weight: 700; }}
    .secondary {{ background: #e2e8f0; color: #0f172a; }}
    pre {{ overflow: auto; background: #020617; color: #f8fafc; padding: 16px; border-radius: 14px; border: 1px solid #334155; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Distributed Incident Response Orchestrator</h1>
    <p>Deterministic incident simulator for OpenEnv-style evaluation.</p>
    <div class="buttons">
      <a class="button secondary" href="/reset">Reset</a>
      <a class="button secondary" href="/state.json">State JSON</a>
    </div>
    <div class="buttons">{buttons}</div>
    <div class="grid">
      <div class="stat"><strong>Incident</strong><div>{state['incident_type']}</div></div>
      <div class="stat"><strong>Severity</strong><div>{state['severity']}</div></div>
      <div class="stat"><strong>System Load</strong><div>{state['system_load']}</div></div>
      <div class="stat"><strong>Time Remaining</strong><div>{state['time_remaining']}</div></div>
      <div class="stat"><strong>Active Incidents</strong><div>{state['active_incidents']}</div></div>
      <div class="stat"><strong>Resources</strong><div>{state['resources_available']}</div></div>
      <div class="stat"><strong>Terminated</strong><div>{state['terminated']}</div></div>
      <div class="stat"><strong>Reason</strong><div>{state['termination_reason']}</div></div>
    </div>
    <h2>Metrics</h2>
    <pre>{json.dumps(state['metrics'], indent=2)}</pre>
  </main>
</body>
</html>"""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler signature
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/state.json":
                self._send_json(env.state().model_dump())
                return

            if parsed.path == "/reset":
                env.reset()
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            if parsed.path == "/step":
                action_name = query.get("action", ["ignore"])[0]
                observation, reward, done, info = env.step(action_name)
                payload = {"observation": observation.model_dump(), "reward": reward, "done": done, "info": info}
                if query.get("format", [""])[0] == "json":
                    self._send_json(payload)
                else:
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                return

            self._send_html(self._render_page())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - stdlib signature
            return

    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Serving incident environment on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    """Entry point used by the Docker container and local runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Run the HTTP interface.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host when using --serve.")
    parser.add_argument("--port", default=7860, type=int, help="Server port when using --serve.")
    parser.add_argument("--seed", default=42, type=int, help="Deterministic seed for the environment.")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port, args.seed)
    else:
        run_demo(args.seed)


if __name__ == "__main__":
    main()
