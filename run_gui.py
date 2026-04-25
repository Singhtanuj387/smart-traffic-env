import time
from smart_traffic.client import SmartTrafficEnv
from smart_traffic.visualization.grid_renderer import GridRenderer

if __name__ == "__main__":
    env = SmartTrafficEnv(base_url="http://localhost:8000")
    renderer = GridRenderer(fps=30)
    
    obs = env.reset(scenario="rush_hour").observation
    
    for _ in range(1000):
        renderer.render(obs)
        # Random light changes (4) or mostly holding (0-3)
        actions = [4] * 81 
        obs = env.step_flat(actions).observation
        time.sleep(0.1)
