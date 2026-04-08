"""Command-line demo and OpenEnv-compatible HTTP server."""

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
    env = IncidentEnvironment(seed=seed)

    print("START")

    state = env.reset()
    print(json.dumps(state.model_dump(), indent=2))

    done = False
    step_count = 0

    while not done:
        action_name = "restart_service"

        observation, reward, done, info = env.step(action_name)

        print(
            json.dumps(
                {
                    "step": step_count,
                    "action": action_name,
                    "reward": reward,
                    "done": done,
                    "state": observation.model_dump(),
                },
                indent=2,
            )
        )

        step_count += 1

    print("END")


def run_server(host: str, port: int, seed: int | None) -> None:
    env = IncidentEnvironment(seed=seed)

    class RequestHandler(BaseHTTPRequestHandler):

        def _send_json(self, payload: dict[str, object], status: int = 200):
            body = json.dumps(payload, indent=2).encode("utf-8")

            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()

            self.wfile.write(body)

        def _send_html(self, body: str):
            encoded = body.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()

            self.wfile.write(encoded)

        # -------------------------
        # REQUIRED POST ENDPOINTS
        # -------------------------

        def do_POST(self):

            parsed = urlparse(self.path)

            # POST /reset
            if parsed.path == "/reset":

                observation = env.reset()

                self._send_json(
                    {
                        "observation": observation.model_dump()
                    }
                )

                return

            # POST /step
            if parsed.path == "/step":

                content_length = int(
                    self.headers.get("Content-Length", 0)
                )

                body = self.rfile.read(content_length)

                try:

                    data = json.loads(body.decode("utf-8"))

                    action_name = data.get(
                        "action",
                        "ignore",
                    )

                except Exception:

                    action_name = "ignore"

                observation, reward, done, info = env.step(
                    action_name
                )

                self._send_json(
                    {
                        "observation": observation.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }
                )

                return

            self._send_json(
                {"error": "Unsupported POST path"},
                status=404,
            )

        # -------------------------
        # REQUIRED GET ENDPOINTS
        # -------------------------

        def do_GET(self):

            parsed = urlparse(self.path)

            query = parse_qs(parsed.query)

            # GET /state
            if parsed.path == "/state":

                self._send_json(
                    env.state().model_dump()
                )

                return

            # Existing JSON endpoint
            if parsed.path == "/state.json":

                self._send_json(
                    env.state().model_dump()
                )

                return

            # UI reset button
            if parsed.path == "/reset":

                env.reset()

                self.send_response(302)

                self.send_header("Location", "/")

                self.end_headers()

                return

            # UI step button
            if parsed.path == "/step":

                action_name = query.get(
                    "action",
                    ["ignore"],
                )[0]

                observation, reward, done, info = env.step(
                    action_name
                )

                payload = {
                    "observation": observation.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }

                if query.get(
                    "format",
                    [""],
                )[0] == "json":

                    self._send_json(payload)

                else:

                    self.send_response(302)

                    self.send_header("Location", "/")

                    self.end_headers()

                return

            # Default UI page
            self._send_html(self._render_page())

        def _render_page(self):

            state = env.state().model_dump()

            buttons = "".join(
                f'<a class="button" href="/step?action={action}">{action}</a>'
                for action in ACTION_BUTTONS
            )

            return f"""
<!doctype html>
<html>
<head>
<title>Distributed Incident Response Orchestrator</title>
</head>
<body>
<h1>Distributed Incident Response Orchestrator</h1>

<a href="/reset">Reset</a>

<div>{buttons}</div>

<pre>
{json.dumps(state, indent=2)}
</pre>

</body>
</html>
"""

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(
        (host, port),
        RequestHandler,
    )

    print(
        f"Serving incident environment on http://{host}:{port}"
    )

    server.serve_forever()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--serve",
        action="store_true",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        default=7860,
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    args = parser.parse_args()

    if args.serve:

        run_server(
            args.host,
            args.port,
            args.seed,
        )

    else:

        run_demo(args.seed)


if __name__ == "__main__":

    main()