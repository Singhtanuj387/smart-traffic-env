import time
from smart_traffic.client import SmartTrafficEnv
from smart_traffic.models import TrafficAction, PHASE_LIST
from smart_traffic.visualization.grid_renderer import GridRenderer

import asyncio

async def main():
    env = SmartTrafficEnv(base_url="http://localhost:8000")
    renderer = GridRenderer()
    
    res = await env.reset(scenario="rush_hour")
    obs = res.observation if hasattr(res, "observation") else res
    
    for _ in range(100):
        renderer.render(obs)
        for agent_id in range(81):
            result = await env.step_single(agent_id, 4)
            obs = result.observation
        time.sleep(0.01)

if __name__ == "__main__":
    asyncio.run(main())
