# Educational Content Generator - Llama 3.2 1B

Fine-tuned Llama 3.2 1B model for generating educational content.

## Model Details
- Base Model: unsloth/Llama-3.2-1B-Instruct
- Training Date: 2025-06-02
- Training Examples: 99
- Model Size: 1B parameters
- Quantization: 4-bit

## Capabilities
- Generate comprehensive study notes
- Create quiz questions with answers
- Produce flashcards for memorization

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llama32_educational_finetuned")
tokenizer = AutoTokenizer.from_pretrained("llama32_educational_finetuned")
```