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

import asyncio

async def main():
    # 1. Configuration
    max_episodes = 5
    max_steps_per_episode = 10  # keep small for example loop
    # Change this to any larger model you have access to!
    # If you want to use "meta-llama/Llama-3.2-1B-Instruct", you MUST visit 
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and click "Agree" first!
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 2. Load Standard HuggingFace Model & Tokenizer
    print("Loading Standard HF Model (Bypassing Unsloth CUDA check)...")
    
    # Force device to CPU since CUDA drivers are out of date
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print(f"Using device: {device}")

    # Grab token from environment variable if it exists
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        padding_side="right",
        token=hf_token
    )
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if device.type == "cuda" else None

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token
    )
    
    if device.type == "cpu":
        base_model = base_model.to(device)
    
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
    
    # Since we are using PEFT/LoRA, we don't need a separate reference model!
    # TRL will automatically disable adapters to compute reference logits, saving 33% VRAM.
    ref_model = None
    
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
    )
    
    try:
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=ppo_model,
            ref_model=None,
            tokenizer=tokenizer,
        )
    except TypeError as e:
        if "unexpected keyword argument 'config'" in str(e):
            print("\n" + "="*80)
            print("CRITICAL ERROR: TRL Version Incompatibility Detected!")
            print("You are using a newer version of TRL (>= 0.10) where online PPO loops via .step() ")
            print("were completely removed by Hugging Face.")
            print("\nTo run this online reinforcement learning script, you MUST downgrade TRL:")
            print("    pip install \"trl<0.9.0\"")
            print("="*80 + "\n")
            import sys; sys.exit(1)
        raise e

    # 3. Setup Environment
    print("Initializing Environment...")
    env_client = SmartTrafficEnv(base_url="http://localhost:8000")

    # 4. Training Loop
    for ep in range(max_episodes):
        print(f"\n--- Episode {ep+1}/{max_episodes} ---")
        
        # Reset environment
        try:
            res = await env_client.reset(scenario="rush_hour")
            obs = res.observation if hasattr(res, "observation") else res
        except Exception as e:
            print(f"Failed to reset environment: {e}")
            break

        queries, responses, rewards = [], [], []

        physics_steps = 0
        while not obs.done and physics_steps < max_steps_per_episode:
            
            tick_queries, tick_responses = [], []
            
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
                    
                    def _gen():
                        return ppo_trainer.generate(
                            [query_tensor],
                            max_new_tokens=2,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9
                        )[0]
                        
                    response_tensor = await asyncio.to_thread(_gen)
                    
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
                step_res = await env_client.step(action_payload)
                obs = step_res.observation
                
                tick_queries.append(query_tensor)
                tick_responses.append(response_only)
                
                # Assign global reward
                if agent_idx == 80:
                    step_reward = step_res.reward if step_res.reward is not None else 0.0
                    
            # Distribute reward across all 81 agents' decisions
            for _ in range(81):
                rewards.append(torch.tensor(step_reward, dtype=torch.float32).to(device))
            
            queries.extend(tick_queries)
            responses.extend(tick_responses)
            
            # Step PPO optimization in chunks of batch_size
            while len(queries) >= ppo_config.batch_size:
                batch_q = queries[:ppo_config.batch_size]
                batch_resp = responses[:ppo_config.batch_size]
                batch_rew = rewards[:ppo_config.batch_size]
                
                def _step():
                    return ppo_trainer.step(batch_q, batch_resp, batch_rew)
                    
                await asyncio.to_thread(_step)
                
                queries = queries[ppo_config.batch_size:]
                responses = responses[ppo_config.batch_size:]
                rewards = rewards[ppo_config.batch_size:]

            physics_steps += 1
            print(f"  Tick {physics_steps}/{max_steps_per_episode} | Reward returned: {step_reward:.4f}")

        # 5. Execute PPO Optimization Step
        print(f"Episode {ep+1} processing complete!")

    print("Training Complete. Saving model...")
    os.makedirs("trl_marl_smart_traffic", exist_ok=True)
    ppo_model.save_pretrained("trl_marl_smart_traffic")
    tokenizer.save_pretrained("trl_marl_smart_traffic")


if __name__ == "__main__":
    asyncio.run(main())
