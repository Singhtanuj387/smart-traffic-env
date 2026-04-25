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
from smart_traffic.models import MultiAgentAction, TrafficAction, PHASE_LIST
from smart_traffic.scenarios.manager import SCENARIO_REGISTRY


class AgentActionSchema(BaseModel):
    actions: List[int] = Field(
        ...,
        description="Exactly 81 integers representing the chosen phase (0-4) for each intersection.",
        min_length=81,
        max_length=81,
    )


def extract_context(obs) -> str:
    """Format the observation into a text state representation."""
    metrics = obs.global_metrics
    flags = obs.scenario_flags

    # Summarize top 5 congested intersections to save token space
    congested_agents = []
    for ag in obs.agents:
        total_q = sum(ag.queue_lengths)
        if total_q > 5:  # Only report if there is traffic
            congested_agents.append((ag.agent_id, total_q, ag.wait_times))
    
    congested_agents.sort(key=lambda x: x[1], reverse=True)
    top_congested = congested_agents[:10]

    context = {
        "step": obs.step,
        "global_metrics": {
            "avg_wait_time": round(metrics.avg_wait_time, 2),
            "total_throughput": metrics.total_throughput,
            "network_efficiency": round(metrics.network_efficiency, 2),
            "congestion_count": metrics.congestion_count
        },
        "active_scenarios": flags.active_scenarios,
        "is_emergency_active": flags.emergency_active,
        "weather_severity": round(flags.weather_severity, 2),
        "top_congested_intersections": [
            {"agent_id": ag[0], "total_queue_ratio": round(ag[1], 2)}
            for ag in top_congested
        ]
    }
    return json.dumps(context, indent=2)


async def run_baseline(server_url: str = "http://localhost:8000", max_steps: int = 100):
    """Run baseline evaluation over all scenarios using OpenAI API / Hugging Face."""
    
    # Check for Hugging Face first
    hf_token = os.environ.get("HF_TOKEN")
    
    # Fallback to standard OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")

    if hf_token:
        print("Using HuggingFace Hub InferenceClient...")
        model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
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
    print(f"Targeting {len(scenarios)} scenarios, max {max_steps} steps per scenario.\n")

    for scenario in scenarios:
        print(f"--- Running scenario: {scenario} ---")
        try:
            res = await env_client.reset(scenario=scenario)
            obs = res.observation if hasattr(res, "observation") else res
            
            total_reward = 0.0
            steps = 0
            
            while not obs.done and steps < max_steps:
                context_str = extract_context(obs)
                
                # Query LLM for phase decisions
                system_prompt = (
                    "You are controlling 81 traffic intersections in a 9x9 grid.\n"
                    "Phases are restricted to [0, 1, 2, 3, 4] mapping to:\n"
                    "0: NS_STRAIGHT, 1: EW_STRAIGHT, 2: NS_LEFT, 3: EW_LEFT, 4: ALL_RED.\n"
                    "Choose an array of exactly 81 phases. You MUST output ONLY a valid raw JSON object representing this schema: {\"actions\": [list of 81 ints]}. "
                    "Do NOT include markdown formatting wrappers like ```json."
                )
                
                if is_hf:
                    # HuggingFace Serverless inferencing
                    response = await llm_client.chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Current State: {context_str}"}
                        ],
                        max_tokens=600,
                        temperature=0.0
                    )
                    raw_content = response.choices[0].message.content
                    try:
                        parsed_actions = json.loads(raw_content)["actions"]
                    except:
                        parsed_actions = [4] * 81
                else:
                    # OpenAI structured parsing
                    response = await llm_client.beta.chat.completions.parse(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Current State: {context_str}"}
                        ],
                        response_format=AgentActionSchema,
                        temperature=0.0
                    )
                    parsed_actions = response.choices[0].message.parsed.actions
                
                # Format for env (Safeguard the output to exactly 81 agents)
                if not isinstance(parsed_actions, list):
                    parsed_actions = [4] * 81
                
                # Truncate and pad
                parsed_actions = parsed_actions[:81]
                if len(parsed_actions) < 81:
                    parsed_actions.extend([4] * (81 - len(parsed_actions)))

                action_batch = MultiAgentAction(actions=[
                    TrafficAction(agent_id=i, phase=PHASE_LIST[a]) 
                    for i, a in enumerate(parsed_actions)
                ])
                
                step_res = await env_client.step(action_batch)
                obs = step_res.observation
                reward = step_res.reward if step_res.reward is not None else 0.0
                
                total_reward += reward
                steps += 1
                
                if steps % 10 == 0:
                    print(f"  Step {steps}/{max_steps} | Instant Reward: {reward:.4f} | Throughput: {obs.global_metrics.total_throughput}")

            avg_reward = total_reward / max(1, steps)
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
    parser = argparse.ArgumentParser(description="Baseline inference script")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="SmartTraffic server URL")
    parser.add_argument("--steps", type=int, default=50, help="Max steps per sequence (to cap API spend)")
    args = parser.parse_args()

    asyncio.run(run_baseline(server_url=args.url, max_steps=args.steps))
