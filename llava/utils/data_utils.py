import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time

# Define special tokens if they are consistent across the project
# These should match what's used in model training and configuration
# DELTA_TOKEN = "<delta_P>"
# WILDTYPE_PROTEIN_TOKEN = "<wt_protein>"
IGNORE_INDEX = -100
WT_PROTEIN_START_TOKEN = "<wt_protein>"
WT_PROTEIN_END_TOKEN = "</wt_protein>"
MUT_PROTEIN_START_TOKEN = "<mut_protein>"
MUT_PROTEIN_END_TOKEN = "</mut_protein>"

class MutationTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_text_len=2048):
        """
        Initialize the dataset.
        Args:
            data_path: Path to the JSON data file
            tokenizer: Pre-configured tokenizer instance
            max_text_len: Maximum text length for tokenization
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        self.max_text_len = max_text_len
        self.samples = self._load_data()
        
        # Debug information
        print(f"[DEBUG] Dataset initialized with:")
        print(f"[DEBUG] - pad_token: {self.tokenizer.pad_token}")
        print(f"[DEBUG] - pad_token_id: {self.tokenizer.pad_token_id}")
        # print(f"[DEBUG] - wt_protein_start_token_id: {self.tokenizer.wt_protein_start_token_id}")
        # print(f"[DEBUG] - wt_protein_end_token_id: {self.tokenizer.wt_protein_end_token_id}")
        # print(f"[DEBUG] - mut_protein_start_token_id: {self.tokenizer.mut_protein_start_token_id}")
        # print(f"[DEBUG] - mut_protein_end_token_id: {self.tokenizer.mut_protein_end_token_id}")
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
            
            # Convert string "None" to actual None
            if mutation_seq == "None":
                mutation_seq = None

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

            print(f"[DEBUG] processed_record: {processed_record}")
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
        mutation_seq = record.get("mutation_seq") # Can be None now
        human_query = record.get("human_query", "")
        gpt_response = record.get("gpt_response", "")

        # Determine attention mode
        has_delta = MUT_PROTEIN_START_TOKEN in human_query
        has_wt = WT_PROTEIN_START_TOKEN in human_query

        print(f"[DEBUG] Sample {idx}:")
        print(f"  - has_delta: {has_delta}")
        print(f"  - has_wt: {has_wt}")
        
        if has_delta and has_wt:
            attention_mode = 'full'
        elif has_delta:
            attention_mode = 'delta_only'
        elif has_wt:
            attention_mode = 'wt_only'
        else:
            # Default or error case if no special tokens are found
            attention_mode = 'text_only'

        print(f"[DEBUG] attention_mode: {attention_mode}")
        print(f"[DEBUG] human_query: {human_query}")

        # Sanitize inputs
        system_prompt = "You are a helpful assistant that explains protein mutations."
        human_query = self._sanitize_text(human_query)
        text_input_for_tokenization = f"{system_prompt}\n\nHuman: {human_query} \n\nAssistant:"
        gpt_response = self._sanitize_text(gpt_response)
        
        # This is the standard way to format inputs for a Causal LM.
        # The model sees the full text, but loss is only calculated on the response part.
        # We also append the EOS token to signal the end of the generation.
        full_text = f"{text_input_for_tokenization}{gpt_response}{self.tokenizer.eos_token}"

        try:
            # Tokenize the full conversation
            tokenized_full = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )
            input_ids = tokenized_full.input_ids.squeeze(0)
            attention_mask = tokenized_full.attention_mask.squeeze(0)

            # Create labels by cloning input_ids and masking the prompt
            labels = input_ids.clone()

            # Tokenize the prompt-only part to find its length for masking
            tokenized_prompt = self.tokenizer(
                text_input_for_tokenization,
                padding=False, # No padding, we just need the length
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )
            prompt_len = tokenized_prompt.input_ids.shape[1]

            # Mask the prompt part of the labels
            labels[:prompt_len] = IGNORE_INDEX
            
            # Also mask padding tokens in the labels
            labels[attention_mask == 0] = IGNORE_INDEX

        except Exception as e:
            print(f"[WARNING] Tokenization failed for idx {idx}: {str(e)}")
            # Return a default/empty item if tokenization fails
            empty_tensor = torch.zeros((self.max_text_len,), dtype=torch.long)
            attention_mask = torch.zeros((self.max_text_len,), dtype=torch.long)
            labels = empty_tensor.clone().fill_(IGNORE_INDEX)
            return {
                "input_ids": empty_tensor,
                "attention_mask": attention_mask,
                "labels": labels,
                "id": record.get("id", f"index_{idx}"),
                "wild_type_sequences": "",
                "mutation_sequences": "",
                "attention_mode": attention_mode,
            }

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "id": record.get("id", f"index_{idx}"),
            "wild_type_sequences": wild_type_seq,
            "mutation_sequences": mutation_seq,
            "attention_mode": attention_mode,
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
        # This list can now contain None values
        collated_batch['mutation_sequences'] = [item['mutation_sequences'] for item in batch]
    
    # Collect attention modes
    if 'attention_mode' in batch[0]:
        collated_batch['attention_mode'] = [item['attention_mode'] for item in batch]
    return collated_batch
