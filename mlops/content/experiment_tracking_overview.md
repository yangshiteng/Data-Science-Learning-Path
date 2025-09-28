# 🧪 What is Experiment Tracking?

👉 In machine learning, an **experiment** = one training run of your model.
It includes:

* **Code** you used 📝
* **Data** you trained on 📊
* **Hyperparameters** (settings) ⚙️
* **Metrics/results** (accuracy, loss, etc.) 📈
* **Model artifact** (the trained model) 🤖

**Experiment tracking** = a system that automatically **records and organizes all this information** so you can compare experiments and reproduce results.

---

# 🎯 Why is it important?

Without experiment tracking:

* You forget which model had the best accuracy.
* You don’t know what hyperparameters gave that result.
* You can’t reproduce yesterday’s success.

With experiment tracking:

* You have a history of every run.
* You can compare experiments side by side.
* You can easily **pick the best model** and deploy it.

---

# 🔑 What You Track

1. **Parameters (inputs)**

   * Example: learning rate = 0.01, batch size = 64.

2. **Metrics (outputs)**

   * Example: accuracy = 85%, loss = 0.4.

3. **Artifacts**

   * The trained model file, plots, confusion matrix.

4. **Code + Data version**

   * Git commit ID + dataset snapshot ID.

5. **Environment**

   * Python/conda/Docker version to make it reproducible.

---

# 🛠️ Popular Tools for Experiment Tracking

* **MLflow** (open-source, most popular).
* **Weights & Biases (W&B)** (cloud-based, user-friendly).
* **Neptune.ai** (focused on team collaboration).
* **Comet.ml** (good visualization dashboards).

---

# 👀 Example: Tracking with MLflow

```python
import mlflow

mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_artifact("confusion_matrix.png")
```

➡️ This logs the parameters, metrics, and artifact. Later you can compare all runs in an MLflow dashboard.

---

# ✨ Simple Analogy

Think of experiment tracking like a **lab notebook 🧑‍🔬**:

* Each time you try a new recipe (model training run), you **write down ingredients (params), process (code/data), and results (metrics)**.
* Later, you flip through your notes and find the **recipe that worked best**.

Without it, you’re just guessing — like a chef who can’t remember how they cooked yesterday’s perfect dish 🍳.

---

# ✅ TL;DR

* **Experiment tracking = keeping a record of every ML training run.**
* It saves **params, metrics, models, and code/data versions**.
* Tools like **MLflow, W&B, Comet** make it easy.
* Benefits: **reproducibility, comparison, best-model selection**.
