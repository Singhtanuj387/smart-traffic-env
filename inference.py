import os
import json
import time
import argparse
import asyncio
from typing import List, Dict, Any

from openai import AsyncOpenAI
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from smart_traffic.client import SmartTrafficEnv
from smart_traffic.models import TrafficAction, PHASE_LIST
from smart_traffic.scenarios.manager import SCENARIO_REGISTRY


def extract_context(obs, active_agent: int) -> str:
    """Format the observation into a text state representation focused on the active agent."""
    metrics = obs.global_metrics
    flags = obs.scenario_flags

    # Active agent's local data
    ag = obs.agents[active_agent]

    # Summarize top 5 congested intersections for global context
    congested_agents = []
    for a in obs.agents:
        total_q = sum(a.queue_lengths)
        if total_q > 5:
            congested_agents.append((a.agent_id, total_q))

    congested_agents.sort(key=lambda x: x[1], reverse=True)
    top_congested = congested_agents[:5]

    context = {
        "step": obs.step,
        "active_agent": active_agent,
        "active_agent_queues": [round(q, 2) for q in ag.queue_lengths],
        "active_agent_current_phase": ag.current_phase,
        "active_agent_neighbor_phases": [round(p, 1) for p in ag.neighbor_phases],
        "global_metrics": {
            "avg_wait_time": round(metrics.avg_wait_time, 2),
            "total_throughput": metrics.total_throughput,
            "network_efficiency": round(metrics.network_efficiency, 2),
            "congestion_count": metrics.congestion_count
        },
        "active_scenarios": flags.active_scenarios,
        "is_emergency_active": flags.emergency_active,
        "top_congested": [
            {"agent_id": c[0], "queue": round(c[1], 2)}
            for c in top_congested
        ]
    }
    return json.dumps(context, indent=2)


async def run_baseline(server_url: str = "http://localhost:8000", max_steps: int = 100):
    """Run baseline evaluation over all scenarios using sequential agent stepping."""

    # Check for Hugging Face first
    hf_token = os.environ.get("HF_TOKEN")

    # Fallback to standard OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")

    if hf_token:
        print("Using HuggingFace Hub InferenceClient...")
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        llm_client = AsyncInferenceClient(model=model_name, token=hf_token)
        is_hf = True
    elif api_key:
        print("Using standard OpenAI API...")
        llm_client = AsyncOpenAI(api_key=api_key)
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        is_hf = False
    else:
        raise ValueError("Either HF_TOKEN or OPENAI_API_KEY environment variable is required.")

    env_client = SmartTrafficEnv(base_url=server_url)

    scenarios = list(SCENARIO_REGISTRY.keys())
    results: Dict[str, float] = {}

    print(f"Starting baseline inference evaluation against {server_url}")
    print(f"Targeting {len(scenarios)} scenarios, max {max_steps} steps per scenario.")
    print(f"Sequential mode: 81 LLM calls per physics step.\n")

    for scenario in scenarios:
        print(f"--- Running scenario: {scenario} ---")
        try:
            res = await env_client.reset(scenario=scenario)
            obs = res.observation if hasattr(res, "observation") else res

            total_reward = 0.0
            physics_steps = 0

            while not obs.done and physics_steps < max_steps:
                # Sequential: step all 81 agents one at a time
                for agent_idx in range(81):
                    context_str = extract_context(obs, active_agent=agent_idx)

                    system_prompt = (
                        f"You are controlling traffic intersection #{agent_idx} in a 9x9 grid.\n"
                        "Phases: 0=NS_STRAIGHT, 1=EW_STRAIGHT, 2=NS_LEFT, 3=EW_LEFT, 4=ALL_RED.\n"
                        "Based on your local queues, neighbor phases, and global metrics, "
                        "choose the best phase (0-4).\n"
                        "Reply with ONLY a single integer (0, 1, 2, 3, or 4). Nothing else."
                    )

                    try:
                        if is_hf:
                            response = await llm_client.chat_completion(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"State: {context_str}"}
                                ],
                                max_tokens=5,
                                temperature=0.0
                            )
                            raw = response.choices[0].message.content.strip()
                            phase = int(raw[0]) if raw and raw[0].isdigit() else 4
                        else:
                            response = await llm_client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"State: {context_str}"}
                                ],
                                max_tokens=5,
                                temperature=0.0
                            )
                            raw = response.choices[0].message.content.strip()
                            phase = int(raw[0]) if raw and raw[0].isdigit() else 4
                    except Exception:
                        phase = 4  # Fallback: ALL_RED

                    phase = max(0, min(4, phase))

                    action = TrafficAction(agent_id=agent_idx, phase=PHASE_LIST[phase])
                    step_res = await env_client.step(action)
                    obs = step_res.observation

                # After all 81 agents: physics ticked, reward is real
                reward = step_res.reward if step_res.reward is not None else 0.0
                total_reward += reward
                physics_steps += 1

                if physics_steps % 5 == 0:
                    print(f"  Step {physics_steps}/{max_steps} | Reward: {reward:.4f} | Throughput: {obs.global_metrics.total_throughput}")

            avg_reward = total_reward / max(1, physics_steps)
            results[scenario] = avg_reward
            print(f"✓ Completed {scenario}: Avg Reward = {avg_reward:.4f}\n")

        except Exception as e:
            print(f"✗ Failed on {scenario}: {e}\n")
            results[scenario] = -999.0

    print("=== BASELINE RESULTS ===")
    overall_score = 0.0
    for scen, score in results.items():
        print(f"  {scen.ljust(25)}: {score:.4f}")
        overall_score += score

    print("-" * 35)
    print(f"  OVERALL AGGREGATE SCORE  : {overall_score / len(scenarios):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline inference script (Sequential)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="SmartTraffic server URL")
    parser.add_argument("--steps", type=int, default=50, help="Max physics steps per scenario")
    args = parser.parse_args()

    asyncio.run(run_baseline(server_url=args.url, max_steps=args.steps))
