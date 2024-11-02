![image](https://github.com/user-attachments/assets/e96340aa-06e9-4c02-9b2b-1eda168ef8bd)

![image](https://github.com/user-attachments/assets/505ed2c2-cb87-401b-9bf3-0a892fad5661)

![image](https://github.com/user-attachments/assets/67ce5f59-cf93-4223-955d-41b531c568b3)

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           random_state=42)

# Create the RFE object and rank each pixel
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5, step=1)
rfe.fit(X, y)

# Selected features
selected_features = pd.DataFrame({'Feature': range(X.shape[1]), 'Ranking': rfe.ranking_})
print(selected_features.sort_values(by="Ranking"))

# Create a model using the selected features
X_transformed = rfe.transform(X)
```
![image](https://github.com/user-attachments/assets/0b75769e-b9f2-4fc3-bcd0-336e8efa6073)
