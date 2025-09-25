# 🧹 What is Data Quality & Validation?

👉 In MLOps, **data is the fuel**. If the fuel is dirty, the engine (your model) won’t run well.
**Data quality & validation** = making sure the data used for training and serving is:

* **Correct** (no obvious mistakes)
* **Consistent** (matches the schema/format)
* **Useful** (relevant and complete enough)

It’s like a **quality check** in a factory 🏭 — you don’t want broken parts (bad data) going into the machine.

---

# 🔑 Why It’s Important

* **Garbage in → garbage out**: Poor data = poor models.
* **Silent errors**: Models may fail subtly if input data changes.
* **Trust**: Good data builds confidence in predictions.
* **Compliance**: Many industries (finance, healthcare) require strict data checks.

---

# ⚠️ Common Data Quality Issues

1. **Missing values**

   * Example: `age = NaN` for some rows.

2. **Wrong types**

   * Example: `"thirty"` instead of `30`.

3. **Outliers**

   * Example: salary = 1,000,000,000.

4. **Duplicates**

   * Same row repeated many times.

5. **Schema drift** (train vs. production mismatch)

   * Training data has `column: age`, but production data changes it to `birth_year`.

6. **Data drift** (distribution changes over time)

   * Training data: mostly young users.
   * Current users: mostly older users.

---

# ✅ What Validation Does

Data validation checks that data **meets expectations** before training or serving.

Examples of checks:

* **Schema check** → Does the dataset have the right columns, types, and ranges?
* **Range check** → Are ages between 0 and 120?
* **Uniqueness check** → Are IDs unique?
* **Distribution check** → Does today’s data look similar to training data?

If a check fails ❌ → alert or block the pipeline.

---

# 🛠️ Tools for Data Validation

* **Great Expectations** (popular open-source)

  * Lets you define “expectations” (rules) like *expect column age between 0–120*.
* **Pandera** (Pythonic data validation for pandas dataframes).
* **TensorFlow Data Validation (TFDV)** (for TF pipelines).
* **WhyLabs / Arize** (monitor data & model quality in production).

---

# 👀 Example with Great Expectations

```python
import great_expectations as ge

df = ge.read_csv("users.csv")

# Check column exists
df.expect_column_to_exist("age")

# Check values within range
df.expect_column_values_to_be_between("age", 0, 120)

# Check no nulls
df.expect_column_values_to_not_be_null("email")
```

➡️ If any test fails, you know the dataset is problematic.

---

# ✨ Analogy

Imagine you run a **restaurant 🍽️**:

* **Ingredients = data**
* **Dish = model**
* If your vegetables are rotten (bad data), no matter how good your recipe is, the dish will taste bad.
* **Validation = kitchen inspection** 👩‍🍳 making sure all ingredients are fresh and in the right quantity before cooking.

---

# 🏁 TL;DR

* **Data quality & validation = making sure data is clean, consistent, and correct.**
* It prevents bad inputs from ruining your models.
* Common checks: missing values, schema consistency, ranges, duplicates, drift.
* Tools: **Great Expectations, Pandera, TFDV, WhyLabs**.
