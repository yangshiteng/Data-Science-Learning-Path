# 🤔 What is Prompting?

**Prompting** = how you talk to an LLM.
It’s just the **instructions you give the model** (the input text), which tells it *what role to play*, *how to answer*, and *in what format*.

👉 Think of it like **asking a person**:

* Vague question → vague answer.
* Clear question with examples → clear, useful answer.

---

# 🛠️ Prompt Engineering

**Prompt Engineering** = the art of writing better prompts so the model gives better results.
It’s not about coding, it’s about **clear instructions + good examples**.

---

# ✨ Key Prompting Techniques

## 1. Role Setting 🎭

Tell the model who it is.

* Example:

  > “You are a friendly teacher. Explain statistics in very simple words.”

---

## 2. Zero-shot Prompting 🚀

Just give the task, no examples.

* Example:

  > “Translate this sentence to French: *How are you today?*”

---

## 3. Few-shot Prompting 🎯

Give examples first, then ask for a new one.

* Example:

  ```
  Q: 2+2  
  A: 4  
  Q: 3+5  
  A: 8  
  Q: 10+12  
  A: ?
  ```

---

## 4. Chain-of-Thought (CoT) 🧩

Ask the model to **think step by step** before answering.

* Example:

  > “Solve this problem step by step: If I have 3 apples and eat 1, how many are left?”

---

## 5. Output Formatting 📄

Tell the model the **exact format** you want.

* Example:

  > “Answer in JSON: {‘answer’: …}”

---

## 6. Instruction Refinement 📝

Be very clear and specific. Avoid vague words.

* Bad ❌: “Summarize this text.”
* Good ✅: “Summarize this text in 3 bullet points, each under 10 words.”

---

# ⚡ Extra Tips for Good Prompts

* Use **simple language** (models can misunderstand complex wording).
* Add **constraints** (word count, style, format).
* Use **examples** for tricky tasks.
* Always **test different prompts** and compare results.

---

# 📚 Example: Same Task, Different Prompts

**Task:** Summarize a news article.

* Vague Prompt:

  > “Summarize this article.”
  > → Long, unfocused summary.

* Better Prompt:

  > “Summarize this article in 3 short bullet points, focusing only on the key events.”
  > → Clear, structured output.

---

# ✅ Summary

* **Prompting** = how you “talk” to an LLM.
* **Prompt Engineering** = writing smart instructions so the model works better.
* Techniques: Role setting 🎭, Zero-shot 🚀, Few-shot 🎯, Chain-of-Thought 🧩, Formatting 📄, Clear instructions 📝.

👉 Good prompts = better answers, less frustration.
