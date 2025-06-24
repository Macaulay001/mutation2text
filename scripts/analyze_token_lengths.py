import json
import numpy as np
from transformers import AutoTokenizer
import os
from tqdm import tqdm

# --- Configuration ---
# Path to your data file
DATA_PATH = 'data/formatted_mutation_data.json'
# The tokenizer used in your training
TOKENIZER_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
# System prompt used in your dataset class
SYSTEM_PROMPT = "You are a helpful assistant that explains protein mutations."
# --- End Configuration ---

def analyze_token_lengths():
    """
    Analyzes the token lengths of the input texts in the dataset to help determine an optimal max_text_len.
    """
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    # It's crucial to use the same tokenizer as in training
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you are logged into huggingface-cli and have access to the model.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer's pad_token to eos_token.")

    print(f"Loading data from: {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    input_lengths = []
    response_lengths = []
    print(f"Analyzing {len(data)} records...")

    for record in tqdm(data, desc="Processing records"):
        conversations = record.get("conversations", [])
        human_query = ""
        gpt_response = ""
        for conv in conversations:
            if conv.get("from") == "human":
                human_query = conv.get("value", "")
            elif conv.get("from") == "gpt":
                gpt_response = conv.get("value", "")

        if not human_query or not gpt_response:
            continue

        # Ensure inputs are strings to prevent tokenization errors
        human_query = str(human_query)
        gpt_response = str(gpt_response)

        # Replicate the exact text construction from your MutationTextDataset
        # We analyze the 'input' part, as that's what consumes most memory in the forward pass.
        # Note: We are not including the special tokens like <delta_P> here as the original query already has them.
        text_input_for_tokenization = f"{SYSTEM_PROMPT}\\n\\nHuman: {human_query} \\n\\nAssistant:"
        
        # Tokenize and get length
        tokenized_input = tokenizer(text_input_for_tokenization, add_special_tokens=True)
        input_lengths.append(len(tokenized_input['input_ids']))
        
        tokenized_response = tokenizer(gpt_response, add_special_tokens=True)
        response_lengths.append(len(tokenized_response['input_ids']))

    if not input_lengths:
        print("No valid records were processed. Please check the data file format.")
        return

    # --- Print Statistics ---
    print("\\n--- Token Length Analysis ---")
    
    print("\\n--- Input Text Lengths (Prompt + Query) ---")
    print(f"Max input token length: {np.max(input_lengths)}")
    print(f"Average input token length: {np.mean(input_lengths):.2f}")
    print(f"90th percentile: {np.percentile(input_lengths, 90):.2f}")
    print(f"95th percentile: {np.percentile(input_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(input_lengths, 99):.2f}")
    print(f"99.9th percentile: {np.percentile(input_lengths, 99.9):.2f}")

    print("\\n--- Response Text Lengths (Assistant's Answer) ---")
    print(f"Max response token length: {np.max(response_lengths)}")
    print(f"Average response token length: {np.mean(response_lengths):.2f}")
    print(f"95th percentile: {np.percentile(response_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(response_lengths, 99):.2f}")

    print("\\nRecommendation:")
    print("Based on the 99th percentile, a `max_text_len` of around "
          f"{int(np.ceil(np.percentile(input_lengths, 99) / 100.0)) * 100} would be a good starting point.")
    print("This would cover the vast majority of your data while being much more memory-efficient than 8192.")

if __name__ == "__main__":
    analyze_token_lengths() 