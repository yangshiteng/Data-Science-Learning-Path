# 🚀 Examples of What You Can Do with Hugging Face Transformers

---

## 🔍 1. **Sentiment Analysis**

Detect the sentiment of a text (positive/negative).

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
print(sentiment("I love using Hugging Face Transformers!"))
```

📌 **Typical models**: `distilbert-base-uncased-finetuned-sst-2-english`

---

## ❓ 2. **Question Answering (QA)**

Answer questions based on a given context.

```python
from transformers import pipeline

qa = pipeline("question-answering")
result = qa(
    question="What is Hugging Face?",
    context="Hugging Face is an open-source company focused on machine learning tools."
)
print(result["answer"])
```

📌 **Typical models**: `bert-large-uncased-whole-word-masking-finetuned-squad`

---

## ✍️ 3. **Text Generation (e.g., Chatbots, Story Writing)**

Generate coherent text completions.

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time, in a distant land,", max_length=50))
```

📌 **Typical models**: `gpt2`, `EleutherAI/gpt-neo-2.7B`, `tiiuae/falcon-7b`

---

## 📝 4. **Text Summarization**

Summarize long documents or articles.

```python
from transformers import pipeline

summarizer = pipeline("summarization")
print(summarizer("The article explains how Transformers revolutionized NLP...")[0]['summary_text'])
```

📌 **Typical models**: `facebook/bart-large-cnn`, `t5-base`

---

## 🌍 5. **Language Translation**

Translate text between languages.

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Hugging Face is amazing!") [0]['translation_text'])
```

📌 **Typical models**: `Helsinki-NLP/opus-mt-*`, `facebook/wmt19-*`

---

## 🧠 6. **Named Entity Recognition (NER)**

Extract names, dates, organizations, etc., from text.

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
print(ner("Apple was founded by Steve Jobs in California."))
```

📌 **Typical models**: `dbmdz/bert-large-cased-finetuned-conll03-english`

---

## 🧑‍💻 7. **Zero-shot Text Classification**

Classify text into arbitrary labels *without training*.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial on Hugging Face.",
    candidate_labels=["education", "entertainment", "finance"]
)
print(result["labels"][0])  # likely "education"
```

📌 **Model**: `facebook/bart-large-mnli`

---

## 🧬 8. **Embeddings / Vector Representations**

Generate embeddings for sentences (for similarity, clustering, etc.)

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Transformers are amazing"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings.shape)  # e.g., torch.Size([1, 384])
```

---

## 🧪 9. **Custom Fine-Tuning / Training**

Use the `Trainer` API to fine-tune models on your dataset.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## 🎨 10. **Multimodal Use (Text + Vision)**

Use CLIP to connect text and images.

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits = outputs.logits_per_image  # similarity score
```

---

## 🔧 Bonus: Using Parameter-Efficient Fine-Tuning (LoRA, Adapters)

If you're using the 🤗 `peft` library:

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=32, lora_dropout=0.1)
peft_model = get_peft_model(model, config)
```

---

## ✅ Summary Table of Use Cases

| Use Case                 | Pipeline                   | Model Example                        |
| ------------------------ | -------------------------- | ------------------------------------ |
| Sentiment Analysis       | `text-classification`      | `distilbert-base-uncased`            |
| Text Generation          | `text-generation`          | `gpt2`, `falcon-7b`                  |
| Summarization            | `summarization`            | `bart-large-cnn`                     |
| Translation              | `translation`              | `opus-mt-en-fr`, `facebook/wmt19`    |
| NER                      | `ner`                      | `bert-base-cased`                    |
| Zero-shot Classification | `zero-shot-classification` | `bart-large-mnli`                    |
| Embeddings               | —                          | `all-MiniLM-L6-v2`                   |
| Vision-Language          | —                          | `openai/clip-vit-base-patch32`       |
| Training                 | —                          | Any `AutoModelFor*` with Trainer API |
