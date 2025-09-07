## 🧰 1. Install Required Packages

If you haven’t already:

```bash
pip install transformers torch
```

---

## 📦 2. Load a Pretrained GPT Model

You can load any GPT model like `gpt2`, `gpt2-medium`, `EleutherAI/gpt-neo-125M`, etc.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

> If you want to try a different variant (like GPT-Neo), change `model_name`.

---

## 🧪 3. Use It for Text Generation

Use the model directly for text generation with the tokenizer:

```python
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
output = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
```

You can customize:

* `max_length`: how long the output should be
* `temperature`: higher = more creative
* `top_k` / `top_p`: control randomness

---

## 💡 Optional: Manual Tokenization and Generation

You can also do it manually for more control:

```python
import torch

input_ids = tokenizer("The sky is", return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## ✅ Summary

| Task           | Code                                   |
| -------------- | -------------------------------------- |
| Load model     | `AutoModelForCausalLM.from_pretrained` |
| Load tokenizer | `AutoTokenizer.from_pretrained`        |
| Generate text  | `pipeline("text-generation")`          |
