import sys
try:
    import _lzma
except ImportError:
    try:
        import backports.lzma
        sys.modules["_lzma"] = backports.lzma
    except ImportError:
        pass

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

# Import our sequential client and logic
from smart_traffic.client import SmartTrafficEnv
from smart_traffic.models import TrafficAction, PHASE_LIST
from inference import extract_context

def main():
    # 1. Configuration
    max_episodes = 5
    max_steps_per_episode = 10  # keep small for example loop
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" # Even smaller model since we are bypassing Unsloth
    
    # 2. Load Standard HuggingFace Model & Tokenizer
    print("Loading Standard HF Model (Bypassing Unsloth CUDA check)...")
    
    # Force device to CPU since CUDA drivers are out of date
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Add LoRA adapters using standard PEFT
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    # TRL PPOTrainer requires a Value Head and a Reference Model
    print("Wrapping model with Value Head for PPO...")
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model).to(device)
    ref_model = create_reference_model(ppo_model).to(device)
    
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
        # Optimize disabled for CPU
        optimize_cuda_cache=False,
    )
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # 3. Setup Environment
    print("Initializing Environment...")
    env_client = SmartTrafficEnv(base_url="http://localhost:8000")

    # 4. Training Loop
    for ep in range(max_episodes):
        print(f"\n--- Episode {ep+1}/{max_episodes} ---")
        
        # Reset environment
        try:
            import asyncio
            res = asyncio.run(env_client.reset(scenario="rush_hour"))
            obs = res.observation if hasattr(res, "observation") else res
        except Exception as e:
            print(f"Failed to reset environment: {e}")
            break

        queries, responses, rewards = [], [], []

        physics_steps = 0
        while not obs.done and physics_steps < max_steps_per_episode:
            
            # Since environment takes 81 sequential sub-steps:
            for agent_idx in range(81):
                # Build context
                state_str = extract_context(obs, active_agent=agent_idx)
                
                # Format prompt for LLM
                prompt = (
                    f"You are controlling traffic intersection #{agent_idx} in a 9x9 grid.\n"
                    "Phases: 0=NS_STRAIGHT, 1=EW_STRAIGHT, 2=NS_LEFT, 3=EW_LEFT, 4=ALL_RED.\n"
                    f"State: {state_str}\n"
                    "Reply with ONLY a single integer (0, 1, 2, 3, or 4)."
                )
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Generate action
                with torch.no_grad():
                    query_tensor = inputs["input_ids"][0]
                    # Generate output
                    response_tensor = ppo_trainer.generate(
                        [query_tensor],
                        max_new_tokens=2,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9
                    )[0]
                    
                    response_only = response_tensor[len(query_tensor):]
                    decoded_response = tokenizer.decode(response_only, skip_special_tokens=True).strip()

                # Parse action (fallback to 4 if invalid)
                try:
                    action_int = int(decoded_response[0]) if decoded_response and decoded_response[0].isdigit() else 4
                    action_int = max(0, min(4, action_int))
                except:
                    action_int = 4
                
                # Step environment sequentially
                action_payload = TrafficAction(agent_id=agent_idx, phase=PHASE_LIST[action_int])
                step_res = asyncio.run(env_client.step(action_payload))
                obs = step_res.observation
                
                # Assign global reward to the batch
                if agent_idx == 80:
                    step_reward = step_res.reward if step_res.reward is not None else 0.0
                    for _ in range(81):
                        rewards.append(torch.tensor(step_reward, dtype=torch.float32).to(device))

                queries.append(query_tensor)
                responses.append(response_only)

            physics_steps += 1
            print(f"  Tick {physics_steps}/{max_steps_per_episode} | Reward returned: {step_res.reward:.4f}")

        # 5. Execute PPO Optimization Step
        print("Optimizing Policy with PPO...")
        stats = ppo_trainer.step(queries, responses, rewards)
        print(f"Episode {ep+1} Loss: {stats['ppo/loss/total']}")

    print("Training Complete. Saving model...")
    ppo_model.save_pretrained("trl_marl_smart_traffic")
    tokenizer.save_pretrained("trl_marl_smart_traffic")


if __name__ == "__main__":
    main()
