# 👀 What is Monitoring & Observability (in ML)?

* **Monitoring** = continuously **watching key signals** (metrics, logs) to catch problems early.
* **Observability** = having **enough visibility** (metrics, logs, traces, samples) to **explain why** something broke and fix it fast.

For ML systems, you monitor **four things**:

1. **System** (is the service healthy?)
2. **Data** (is the input still what we expect?)
3. **Model** (are predictions still good?)
4. **Business** (are we improving user outcomes/KPIs?)

Think of it like **operating a restaurant**:

* System = kitchen equipment works (uptime, latency).
* Data = ingredients are fresh and consistent (schema, ranges).
* Model = taste is right (accuracy, calibration, drift).
* Business = customers are happy (retention, conversions).

---

## 🎛️ What to Monitor (The ML “Golden Four”)

### 1) System Health (DevOps layer)

* **Latency** (p50/p95), **error rate** (4xx/5xx), **throughput**, **CPU/GPU**, **memory**, **cost**.
* Tools: **Prometheus/Grafana**, CloudWatch/Stackdriver, OpenTelemetry.

### 2) Data Quality & Drift

* **Schema parity** (columns, types, ranges, missing/NaNs).
* **Feature statistics** vs. training baseline (means, std, categories).
* **Drift** between training and live data (e.g., PSI, KS test, Jensen–Shannon—don’t worry about formulas; tools compute them).
* Tools: **Great Expectations**, **Evidently**, **WhyLabs**, **Arize**, **TFDV**.

### 3) Model Performance

* **With labels available**: accuracy, AUC, F1, RMSE, calibration (pred prob vs. actual).
* **With delayed/no labels**: proxy metrics (click-through, dwell time), **confidence distributions**, **prediction score drift**, **population mix**, **rejection/abstain rates**.
* Extras: **fairness** (per-group metrics), **explanations** (SHAP distributions for sanity).

### 4) Business Impact

* Conversion rate, revenue per user, retention, support tickets, fraud losses, SLA/SLO compliance.

---

## 🧭 Typical Problems Monitoring Catches

* **Schema break**: new column name; type changed from int → string.
* **Upstream data bug**: suddenly 30% missing values in `age`.
* **Data drift**: users in a new country → very different distribution.
* **Concept drift**: user behavior changes (seasonality, promo), model accuracy drops.
* **Skew**: training preprocessing ≠ serving preprocessing.
* **Infra**: latency spikes, OOM, GPU starvation.
* **Ethics**: performance degrades on a user subgroup.

---

## 🧰 Core Tooling (starter picks)

* **Metrics & dashboards**: Prometheus + Grafana (system), **Evidently/WhyLabs/Arize** (data & model).
* **Logging**: JSON structured logs to ELK/EFK (Elasticsearch/OpenSearch + Kibana).
* **Tracing**: OpenTelemetry for request traces (great when services call other services).
* **Tracking/registry**: MLflow or W&B to tie live model → training lineage.

---

## 🧪 How to Set It Up (step-by-step)

### Step 1: Define SLIs/SLOs (what “good” looks like)

* **System SLOs**: p95 latency < 200ms; error rate < 1%.
* **Model SLOs**: AUC ≥ 0.85 (weekly), drift PSI < 0.2 (daily).
* **Business SLOs**: conversion ≥ baseline − 2% (weekly).

> **Tip:** start with few, clear SLOs. Expand later.

### Step 2: Log the right data at inference

Log (without PII unless allowed): **timestamp, model_version, features (hashed/summary), prediction, confidence, user segment, request_id**.
If labels arrive later, log them with the same `request_id` to join.

**Example (FastAPI + JSON logs)**

```python
from fastapi import FastAPI
import json, time, uuid

app = FastAPI()
MODEL_VERSION = "churn_xgb:3"

@app.post("/predict")
def predict(payload: dict):
    req_id = str(uuid.uuid4())
    start = time.time()
    x = extract_features(payload)           # keep schema consistent!
    y_hat, p = model_predict(x)             # prediction + confidence
    latency_ms = (time.time() - start)*1000
    log = {
        "ts": time.time(),
        "request_id": req_id,
        "model_version": MODEL_VERSION,
        "features_summary": feature_summary(x),  # e.g., bucketed/hashed
        "prediction": int(y_hat),
        "confidence": float(p),
        "latency_ms": latency_ms
    }
    print(json.dumps(log))                  # ship to your log pipeline
    return {"request_id": req_id, "prediction": int(y_hat), "confidence": p}
```

### Step 3: Compute daily/weekly reports

* Build a small job (Airflow/Prefect) that:

  * Pulls yesterday’s prediction logs (+ labels if available).
  * Computes **data profile** & **drift** vs. training.
  * Computes **performance** per segment.
  * Sends **alerts** if thresholds are breached.

**Using Evidently (conceptual)**

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df_sample, current_data=yesterday_df)
report.save_html("reports/data_drift_2025-09-27.html")
```

### Step 4: Dashboards & alerts

* Grafana: system graphs (latency, errors, QPS).
* Model dashboard: drift over time, top features drifting, performance by segment.
* Alerts: Slack/Email/PagerDuty for **violations** (e.g., PSI > 0.2, p95 latency > SLO).

### Step 5: Response playbooks

Document what to do when:

* **Data drift high** → check upstream pipeline; hotfix mapping; consider retrain.
* **Latency spike** → autoscale, profile model, quantize/optimize.
* **Accuracy drop** → compare to shadow/canary model; rollback if needed.
* **Fairness issue** → enable per-group calibration or thresholds; investigate features.

---

## 🧪 Canary, Shadow, and A/B (safe rollout patterns)

* **Shadow**: new model gets the same traffic **but its predictions aren’t served**—you just log & compare. Zero risk.
* **Canary**: send **small% of traffic** (e.g., 5%) to the new model. If metrics OK → ramp up.
* **A/B test**: split traffic and compare business outcomes statistically.

These patterns + monitoring = safe, data-driven deployment decisions.

---

## 🕰️ What if Labels Arrive Late (common in production)?

* Use **proxy/leading indicators**: confidence distribution, score drift, rejection rate, CTR, bounce rate.
* When labels arrive (daily/weekly), **backfill** performance metrics and update dashboards.
* Keep **“label delay”** visible so stakeholders know interpretation limits.

---

## 🧑‍⚖️ Fairness, Privacy, and Compliance

* Track performance **per segment** (region, device, age group if allowed).
* Monitor **calibration** (predicted probabilities match real outcomes).
* **PII**: hash/perturb features; enforce access controls; keep audit logs.
* Keep **model cards** with known limits and monitoring plan.

---

## 📋 Minimal Monitoring Checklist (copy/paste)

* [ ] **System**: p95 latency, error rate, QPS, CPU/GPU, memory, cost
* [ ] **Data**: schema checks, missing/NaN, category coverage, range checks
* [ ] **Drift**: feature drift (PSI/KS), prediction-score drift vs. training
* [ ] **Model**: accuracy/AUC/RMSE (when labels available), calibration, per-segment metrics
* [ ] **Business**: conversion/revenue/retention deltas vs. baseline
* [ ] **Alerts**: thresholds + clear on-call playbooks
* [ ] **Lineage**: live **model_version** tied to training run (commit, data snapshot)
* [ ] **Dashboards**: grafana (system) + model monitor (data/model)
* [ ] **Rollout safety**: shadow/canary + rollback button

---

## 🧠 Quick “Good Enough” Starter Setup

* **Prometheus + Grafana** → system SLOs (latency/errors).
* **Evidently** (open-source) → daily drift + performance report.
* **MLflow** → map **production model version** back to training run & data snapshot.
* **Prefect** → schedule daily monitoring jobs and send Slack alerts.

That stack gets you robust basics without heavy complexity.

---

### TL;DR

Monitoring & Observability = **seeing problems early and understanding them quickly**.
Watch **system health, data quality, model performance, and business impact**; add **alerts and playbooks**; roll out with **shadow/canary** for safety; tie everything to **model lineage** so you can debug and fix fast.
