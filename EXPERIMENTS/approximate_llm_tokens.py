import os
import json
import tiktoken # pip install tiktoken

def main():
    # Load the standard tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # 1. Calculate System Prompt Tokens
    system_prompt_path = "system_prompt.md"
    definitions_path = "definitions.txt"
    
    sys_tokens = 0
    if os.path.exists(system_prompt_path) and os.path.exists(definitions_path):
        with open(system_prompt_path, "r") as f: sys_text = f.read()
        with open(definitions_path, "r") as f: def_text = f.read()
        # Rough recreation of your prompt injection
        full_sys_prompt = sys_text.replace("[TAXONOMY - DEFINITIONS - RULES]", def_text)
        sys_tokens = len(enc.encode(full_sys_prompt))
        print(f"✅ System Prompt Size: {sys_tokens} tokens")
    else:
        print("⚠️ Could not find system_prompt.md or definitions.txt. Assuming ~1200 tokens.")
        sys_tokens = 1200

    # 2. Calculate Input/Output Tokens from existing JSONL
    llm_file = "RESULTS/LLM_PREDICTIONS/ner_results_gemini_3_pro_preview.jsonl"
    
    if not os.path.exists(llm_file):
        print(f"❌ Could not find {llm_file}.")
        return

    total_input_tokens = 0
    total_output_tokens = 0
    num_success = 0
    num_total = 0

    with open(llm_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            task = json.loads(line)
            num_total += 1

            sentence = task.get("input_text", "")
            total_input_tokens += sys_tokens + len(enc.encode(sentence))  # every attempt costs input

            response = task.get("raw_llm_response", "")
            if response:
                total_output_tokens += len(enc.encode(response))
                num_success += 1

    cost_to_obtain_success = (total_input_tokens/1e6)*2.00 + (total_output_tokens/1e6)*12.00
    cost_per_successful_sentence = cost_to_obtain_success / num_success
    total_cost = cost_per_successful_sentence * 28_400_000

    print(f"Attempts: {num_total}, Successes: {num_success}, Retry rate: {(num_total-num_success)/num_total:.1%}")
    print(f"Estimated cost for 28.4M sentences: ${total_cost:,.2f}")

if __name__ == "__main__":
    main()