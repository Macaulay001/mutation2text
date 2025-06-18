import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time

# Define special tokens if they are consistent across the project
# These should match what's used in model training and configuration
DELTA_TOKEN = "<delta_P>"
IGNORE_INDEX = -100

class MutationTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_text_len=512, require_both_sequences=True):
        """
        Initialize the dataset.
        Args:
            data_path: Path to the JSON data file
            tokenizer: Pre-configured tokenizer instance
            max_text_len: Maximum text length for tokenization
            require_both_sequences: Whether both wild type and mutation sequences are required
        """
        self.data_path = data_path
        self.tokenizer = tokenizer  # Use the pre-configured tokenizer
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            print("[DEBUG] Setting pad_token to eos_token in dataset")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add the special delta token if it's not already part of the tokenizer
        if DELTA_TOKEN not in self.tokenizer.additional_special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [DELTA_TOKEN]})
            # The model's token embeddings will need to be resized accordingly
        
        self.delta_token_id = self.tokenizer.convert_tokens_to_ids(DELTA_TOKEN)
        self.max_text_len = max_text_len
        self.require_both_sequences = require_both_sequences
        self.samples = self._load_data()
        
        # Debug information
        print(f"[DEBUG] Dataset initialized with:")
        print(f"[DEBUG] - pad_token: {self.tokenizer.pad_token}")
        print(f"[DEBUG] - pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"[DEBUG] - delta_token_id: {self.delta_token_id}")
        print(f"[DEBUG] - max_text_len: {self.max_text_len}")
        print(f"[DEBUG] - num_samples: {len(self.samples)}")

    def _load_data(self):
        samples = []
        try:
            with open(self.data_path, 'r') as f:
                raw_data_list = json.load(f) # Load the entire JSON list
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return samples
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.data_path}: {e}")
            return samples

        for item_from_list in raw_data_list:
            wild_type_seq = item_from_list.get("wild_type_seq")
            mutation_seq = item_from_list.get("mutation_seq")
            conversations = item_from_list.get("conversations", [])

            human_query = ""
            gpt_response = ""

            if isinstance(conversations, list):
                for conv_item in conversations:
                    if isinstance(conv_item, dict):
                        if conv_item.get("from") == "human":
                            human_query = conv_item.get("value", "")
                        elif conv_item.get("from") == "gpt":
                            gpt_response = conv_item.get("value", "")

            # Basic filtering (optional, can be done upstream or based on flags)
            if self.require_both_sequences and (not wild_type_seq or not mutation_seq):
                continue
            # We now require both human_query and gpt_response to be present
            if not human_query or not gpt_response:
                 print(f"Warning: Record {item_from_list.get('id', 'Unknown ID')} is missing human query or gpt response.")
                 continue # Skip records without both human query and gpt response
            
            processed_record = {
                "wild_type_seq": wild_type_seq,
                "mutation_seq": mutation_seq,
                "human_query": human_query,
                "gpt_response": gpt_response,
                "id": item_from_list.get("id") # Preserve ID if present, for debugging
            }
            samples.append(processed_record)
        return samples

    def _sanitize_text(self, text):
        """
        Sanitize text input for tokenization.
        """
        if not isinstance(text, str):
            text = str(text)
        # Replace any potential problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t0 = time.time()
        record = self.samples[idx]
        wild_type_seq = record.get("wild_type_seq", "")
        mutation_seq = record.get("mutation_seq", "")
        human_query = record.get("human_query", "") # Get the original human query
        gpt_response = record.get("gpt_response", "")

        # Sanitize inputs
        system_prompt = "You are a helpful assistant that explains protein mutations."
        human_query = self._sanitize_text(human_query)
        processed_query = human_query.replace("<wt_prot_seq> <mut_prot_seq>", DELTA_TOKEN)
        # print(f"[DEBUG] Processed query for idx {idx}: {processed_query}")
        text_input_for_tokenization = f"{system_prompt}\n\nHuman: {processed_query} \n\nAssistant:"
        # text_input_for_tokenization = self._sanitize_text(human_query.replace("<wt_prot_seq>\n<mut_prot_seq>", DELTA_TOKEN))
        gpt_response = self._sanitize_text(gpt_response)

        # print(f"[DEBUG] Tokenizing input: {text_input_for_tokenization}: gpt_response: {gpt_response}")
        
        try:
            # Tokenize the prepared text input
            tokenized_input = self.tokenizer(
                text_input_for_tokenization,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            # Tokenize gpt response for labels
            tokenized_label = self.tokenizer(
                gpt_response,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )

        except Exception as e:
            print(f"[ERROR] Tokenization failed for idx {idx}: {str(e)}")
            # Return a default/empty item if tokenization fails
            empty_tensor = torch.zeros((self.max_text_len,), dtype=torch.long)
            return {
                "input_ids": empty_tensor,
                "attention_mask": empty_tensor,
                "labels": empty_tensor.fill_(IGNORE_INDEX),
                "id": record.get("id", f"index_{idx}"),
                "wild_type_sequences": "",
                "mutation_sequences": ""
            }

        input_ids = tokenized_input.input_ids.squeeze(0)
        unique_tokens = torch.unique(input_ids)
        # print(f"[DEBUG] Unique tokens in input for idx {idx}: {unique_tokens.tolist()}")


        attention_mask = tokenized_input.attention_mask.squeeze(0)
        labels = tokenized_label.input_ids.squeeze(0)

        # Set labels to IGNORE_INDEX for padding tokens in the label sequence
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "id": record.get("id", f"index_{idx}"),
            # Include protein sequences separately for the model's protein encoder
            "wild_type_sequences": wild_type_seq,
            "mutation_sequences": mutation_seq,
        }

        t1 = time.time()
        # print(f"[TIME] __getitem__ index {idx} took {t1-t0:.4f} sec")
        return item

def custom_collate_fn(batch):
    t0 = time.time()
    """
    Custom collate function to handle batching of sequences and text.
    Assumes that the model itself will handle the merging of protein and text embeddings.
    This collate function primarily batches the text token IDs, attention masks, and labels,
    and collects the protein sequences as lists.
    """
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    # Stack numerical tensors
    # Padding has already been applied in __getitem__ to a fixed max_text_len
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)

    collated_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    # Collect protein sequences if present
    if 'wild_type_sequences' in batch[0]:
        collated_batch['wild_type_sequences'] = [item['wild_type_sequences'] for item in batch]
    if 'mutation_sequences' in batch[0]:
        collated_batch['mutation_sequences'] = [item['mutation_sequences'] for item in batch]
    # t1 = time.time()
    # print(f"[TIME] custom_collate_fn for batch of size {len(batch)} took {t1-t0:.4f} sec")
    return collated_batch

# Example of how to get delta_token_id for the model:
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# tokenizer.add_special_tokens({'additional_special_tokens': [DELTA_TOKEN]})
# DELTA_TOKEN_ID = tokenizer.convert_tokens_to_ids(DELTA_TOKEN)
# model.set_delta_token_id(DELTA_TOKEN_ID) # Assuming a setter in your LlavaLlamaForCausalLM
# model.resize_token_embeddings(len(tokenizer)) # Important!