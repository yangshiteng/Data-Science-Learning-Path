# 🛠️ Tools for Versioning in MLOps

## 1. **Code 📝**

👉 You already know this one:

* **Git** → the standard for code versioning.
* **GitHub / GitLab / Bitbucket** → platforms to host and collaborate.

💡 Example: save your training scripts and configs with Git commits/tags.

---

## 2. **Data 📊**

Data can be huge (GBs–TBs), so we need special tools:

* **DVC (Data Version Control)** → Git-like tool for large datasets, works with cloud storage (S3, GCS, Azure).
* **Git LFS (Large File Storage)** → good for medium-sized data like images/audio.
* **Lakehouse formats** → Delta Lake, Apache Iceberg, Hudi (if using big data infrastructure) → allow “time travel” on tables.
* **Feature Stores** → Feast (store and version engineered features).

💡 Example: store “customer\_data\_v1.csv” in DVC with a unique version hash.

---

## 3. **Models 🤖**

Trained models are artifacts → you want a **model registry**.

* **MLflow Model Registry** → open-source, widely used.
* **Weights & Biases (W\&B) Artifacts** → version models and datasets.
* **SageMaker Model Registry (AWS)**, **Vertex AI (GCP)**, **Azure ML** → cloud-native registries.
* **BentoML** → packages and versions models for deployment.

💡 Example: register `churn_model:v2` and promote it from *staging* → *production*.

---

## 4. **Environment ⚙️**

Need to lock down dependencies so things don’t break.

* **requirements.txt / poetry.lock / conda.yaml** → capture Python packages.
* **Docker** → containerize everything (OS + libraries + model).
* **Nix** (advanced) → reproducible environments.

💡 Example: build a Docker image `my_model:1.0` with Python 3.9 + sklearn 1.2.

---

# 🔗 Putting It Together

| What to version | Tools                                                    |
| --------------- | -------------------------------------------------------- |
| Code            | Git, GitHub/GitLab                                       |
| Data            | DVC, Git LFS, Delta Lake, Feast                          |
| Models          | MLflow, W\&B, SageMaker/Vertex/Azure registries, BentoML |
| Environment     | Conda, Pip/Poetry, Docker, Nix                           |

---

# ✅ TL;DR

* **Code** → Git
* **Data** → DVC / Git LFS / Lakehouse formats
* **Models** → MLflow / W\&B / Cloud registries
* **Environment** → Conda / Docker

👉 Together, these ensure you can **reproduce, trace, and safely roll back** any ML experiment.
