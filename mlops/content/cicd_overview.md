# 🔄 What is Continuous Training (CT)?

👉 Continuous Training = **automatically retraining models when needed** (new data, drift, or performance drops).

Why?

* Data changes over time (user behavior, market trends, seasonality).
* A model trained once may become outdated.
* CT ensures the model **stays fresh and relevant**.

**Example**:

* You have a fraud detection model.
* Fraudsters change tactics → old model fails.
* CT pipeline picks up new transactions, retrains weekly, redeploys if accuracy improves.

---

# ⚙️ What is CI/CD in MLOps?

CI/CD comes from software engineering but adapted for ML:

* **CI (Continuous Integration)** → automatically test and validate changes to **code, data, and models**.
* **CD (Continuous Deployment/Delivery)** → automatically deliver new models to staging or production if they pass tests.

**Difference from normal software CI/CD:**

* In ML, we don’t just test code. We also test **data quality** and **model performance**.

---

# 🧩 Typical Continuous Training & CI/CD Workflow

1. **Data arrives** (daily/weekly or event-driven).
2. **Pipeline runs**:

   * Data validation (schemas, missing values).
   * Feature engineering.
   * Model training.
   * Model evaluation.
3. **Automated tests**:

   * Unit tests (does code run?).
   * Data tests (is schema correct?).
   * Performance tests (is accuracy ≥ baseline?).
4. **Registry update**: best model is versioned & stored.
5. **Deployment**:

   * If metrics pass → promote to staging/production.
   * Canary/shadow deployment for safety.
6. **Monitoring**:

   * Watch live performance → trigger retraining if needed.

---

# 🔔 Triggering Continuous Training

CT can be triggered by:

* **Time-based** (e.g., retrain daily/weekly).
* **Data-based** (when enough new data is collected).
* **Performance-based** (when accuracy drops below threshold).

---

# 🛠️ Tools You’ll See

* **CI/CD tools**: GitHub Actions, GitLab CI, Jenkins.
* **Pipeline/orchestration**: Prefect, Airflow, Dagster, Kubeflow.
* **Model registries**: MLflow, W&B, SageMaker Model Registry.
* **Deployment**: BentoML, KServe, Vertex AI, SageMaker Endpoints.

---

# 👀 Example (Simplified Flow)

1. Push new training code → GitHub Actions runs tests.
2. Prefect/Airflow job retrains model with new data.
3. MLflow logs model + metrics.
4. If metrics > current prod model → auto-deploy via Docker/Kubernetes.
5. Grafana/WhyLabs monitors drift → triggers retrain if drift > threshold.

---

# ✨ Analogy

Think of your model like a **self-driving car 🚗**:

* **CI/CD** = mechanics checking the engine, brakes, and sensors before every trip (code/data/model tests).
* **Continuous Training** = updating the GPS maps regularly so the car doesn’t get lost (retraining with new data).

Together, they ensure the car runs smoothly and stays up-to-date.

---

# ✅ TL;DR

* **Continuous Training (CT)** → keep models fresh by retraining automatically when new data arrives or performance drops.
* **CI/CD in ML** → automatically test and deploy models, ensuring safe, reliable updates.
* Benefits: **automation, reproducibility, faster delivery, fewer errors**.
