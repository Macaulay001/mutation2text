# __init__.py for llava.model

# Attempt to make key components available when the package is imported.
# This depends on the final class names and their relevance for direct import.

# try:
#     from .llava_arch import LlavaLlamaForCausalLM # Example if this is the main model class
#     from .esm_protein_encoder import ESMProteinEncoder
# except ImportError as e:
#     print(f"Error importing in llava.model.__init__: {e}. This might be due to dependencies not yet created or circular imports during initial setup.")

# It's often safer to let users import directly from the submodules, e.g.:
# from llava.model.llava_arch import LlavaLlamaForCausalLM

# print("llava.model package initialized") # For debugging, can be removed 