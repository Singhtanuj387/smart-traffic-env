import os
import argparse
import asyncio
import numpy as np
from smart_traffic.client import SmartTrafficEnv
from smart_traffic.training.gym_adapter import TrafficGymAdapter
from smart_traffic.training.mappo import MAPPO

def main():
    parser = argparse.ArgumentParser(description="Train MAPPO against the Smart Traffic Environment")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="SmartTraffic server URL")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    args = parser.parse_args()

    print("Connecting to Smart Traffic Server...")
    # OpenEnv requires Async connections or we can use the GymAdapter which is synchronous wrapper
    # The gym adapter takes the base URL and handles the async client internally
    env = TrafficGymAdapter(server_url=args.url)
    
    trainer = MAPPO(obs_dim=47, n_actions=5, n_agents=81, lr=3e-4)

    print(f"Starting Training for {args.episodes} episodes...")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 200:
            actions, log_probs, values = trainer.act(obs)
            
            next_obs, global_reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            per_agent_rewards = np.array(info.get("per_agent_rewards", [global_reward] * trainer.n_agents))
            
            # Record trajectory
            trainer.buffer.add(
                obs=obs,
                action=actions,
                log_prob=log_probs,
                reward=per_agent_rewards,
                value=values,
                done=done
            )
            
            obs = next_obs
            total_reward += global_reward
            steps += 1
            
        # Run PPO Update at the end of the episode
        metrics = trainer.update()
        
        loss = metrics.get('loss', 0.0)
        print(f"Episode {ep:03d} | Steps: {steps} | Total Reward: {total_reward:0.3f} | Loss: {loss:0.4f}")

    print("Training Complete. Saving model...")
    os.makedirs("weights", exist_ok=True)
    trainer.save("weights/mappo_checkpoint.pt")
    
    # Properly shutdown networking 
    env.close()

if __name__ == "__main__":
    main()
