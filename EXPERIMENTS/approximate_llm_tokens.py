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
    num_sentences = 0
    
    with open(llm_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            task = json.loads(line)
            
            # Input = System Prompt + Sentence
            sentence = task.get("input_text", "")
            sent_tokens = len(enc.encode(sentence))
            total_input_tokens += (sys_tokens + sent_tokens)
            
            # Output = Raw JSON Response
            response = task.get("raw_llm_response", "")
            if response:
                total_output_tokens += len(enc.encode(response))
            
            num_sentences += 1
            
    avg_input = total_input_tokens / num_sentences
    avg_output = total_output_tokens / num_sentences
    
    print("\n" + "="*50)
    print(f"📊 GEMINI 3.0 PRO TOKEN APPROXIMATION (Over {num_sentences} sentences)")
    print("="*50)
    print(f"Average Input Tokens per Call:  {avg_input:.0f}")
    print(f"Average Output Tokens per Call: {avg_output:.0f}")
    print(f"\nCost per 1M Input: $2.00 | Cost per 1M Output: $12.00")
    
    # Calculate projection for 28.4 Million sentences
    cost_input = (avg_input / 1_000_000) * 2.00 * 28_400_000
    cost_output = (avg_output / 1_000_000) * 12.00 * 28_400_000
    total_cost = cost_input + cost_output
    
    print(f"Estimated Cost for 28.4M sentences: ${total_cost:,.2f}")
    print("="*50)

if __name__ == "__main__":
    main()