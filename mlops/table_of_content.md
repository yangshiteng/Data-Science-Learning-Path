## **[1. What is MLOps?]()**
---

## **2. Versioning**

* Code versioning (Git basics)
* Data versioning (DVC, Git LFS)
* Model versioning (model registries)

---

## **3. Experiment Tracking**

* Why “notebooks only” isn’t enough
* Parameters, metrics, artifacts, logs
* Tools: MLflow, Weights & Biases (W\&B)

---

## **4. Data Quality & Validation**

* Data drift vs. concept drift
* Schema consistency (train vs. serve)
* Tools: Great Expectations, Pandera

---

## **5. Pipelines & Orchestration**

* Why ML pipelines are DAGs (data → train → eval → deploy)
* Orchestration tools: Prefect, Airflow, Dagster
* Scheduling and automation

---

## **6. Model Packaging**

* Why packaging matters (train env ≠ prod env)
* Formats: Pickle, ONNX, TorchScript
* Containerization with Docker

---

## **7. Model Deployment**

* Deployment types: batch, real-time API, streaming
* Deployment strategies: blue/green, canary, shadow
* Serving tools: FastAPI, BentoML, TF Serving, TorchServe

---

## **8. Monitoring & Observability**

* Metrics: latency, throughput, error rate
* Model monitoring: drift, performance decay
* Tools: Prometheus, Grafana, Arize/WhyLabs

---

## **9. Continuous Training & CI/CD**

* What is continuous training (CT) vs. continuous deployment (CD)
* When/why to retrain models
* CI/CD basics for ML (tests, build, deploy)

---

## **10. Governance & Responsible AI**

* Model lineage & reproducibility
* Access control, approvals, audit logs
* Ethics: fairness, explainability, bias checks
