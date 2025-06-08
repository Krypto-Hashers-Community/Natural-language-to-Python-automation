from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_code(nl_prompt, max_length=128):
    input_text = f"Generate Python code: {nl_prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompts = [
    "Write a function that calculates the factorial of a number",
    "Create a list comprehension that filters odd numbers from 1 to 20",
    "Write a simple for loop"
]

for p in prompts:
    print("📝 Prompt:", p)
    print("🐍 Code:\n", generate_code(p))
    print("-" * 50)

#==========================================================
#!huggingface-cli login #do this if needed
#==========================================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model (StarCoderBase is recommended for general use)
model_id = "bigcode/starcoderbase"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

def generate_code(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompts
prompts = [
    "# Write a function that calculates the factorial of a number\n",
    "# Create a list comprehension that filters odd numbers from 1 to 20\n",
    "# Write a simple for loop\n"
]

for p in prompts:
    print("📝 Prompt:\n", p)
    print("🐍 Generated Code:\n", generate_code(p))
    print("-" * 60)
