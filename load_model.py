"""
This script loads the Meta-Llama-3-8B language model using 
Hugging Face's Transformers library. It also loads the BLIP-2 vision 
encoder and processor for multimodal input handling.

- LLaMA 3 (meta-llama/Meta-Llama-3-8B) loaded with bfloat16 for GPU use.
- BLIP-2 (Salesforce/blip2-opt-2.7b) used as the vision model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, Blip2Processor, Blip2Model
import torch

ID_LLAMA3_8B = "meta-llama/Meta-Llama-3-8B"
ID_BLIP2_27B = "Salesforce/blip2-opt-2.7b"

def load_model(token: str=None):
    
    ## LLM
    tokenizer = AutoTokenizer.from_pretrained(ID_LLAMA3_8B, token = token)
    model = AutoModelForCausalLM.from_pretrained(
        ID_LLAMA3_8B,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        token = token
    )
    
    # Vision
    blip_processor = Blip2Processor.from_pretrained(ID_BLIP2_27B)
    blip_model = Blip2Model.from_pretrained(ID_BLIP2_27B).to('cuda')
    
    return tokenizer, model, blip_processor, blip_model
