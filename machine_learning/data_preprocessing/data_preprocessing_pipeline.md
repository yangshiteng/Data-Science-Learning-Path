![image](https://github.com/user-attachments/assets/b714c49d-cd83-4f0e-91d0-482938b28995)

![image](https://github.com/user-attachments/assets/f3bed853-a2d6-4bbf-a30a-9917e974be72)

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample data
data = pd.DataFrame({
    'age': [25, 27, 29, None, 34],
    'salary': [50000, 48000, 54000, 52000, None],
    'state': ['CA', 'CA', 'NY', 'NY', 'TX'],
    'purchased': [0, 1, 0, 1, 1]
})

# Separate features and target
X = data.drop('purchased', axis=1)
y = data['purchased']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['age', 'salary']),
        ('cat', categorical_transformer, ['state'])
    ])

# Creating a full pipeline with classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Fitting the pipeline
pipeline.fit(X_train, y_train)

# Predicting with the pipeline
predictions = pipeline.predict(X_test)
```
![image](https://github.com/user-attachments/assets/f7001132-47ad-4233-bf20-7459838d6b74)








