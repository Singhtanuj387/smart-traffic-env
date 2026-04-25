# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smart Traffic Environment.

Exposes the SmartTrafficEnvironment over HTTP and WebSocket endpoints.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

import os
import sys

# Ensure direct execution works by adding the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'pip install openenv-core[core]'"
    ) from e

from smart_traffic.models import MultiAgentAction, TrafficObservation
from smart_traffic.server.smart_traffic_environment import SmartTrafficEnvironment


# Create the app with openenv HTTP server wrapper
app = create_app(
    SmartTrafficEnvironment,
    MultiAgentAction,
    TrafficObservation,
    env_name="smart-traffic-env",
    max_concurrent_envs=4,  # support parallel training sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
