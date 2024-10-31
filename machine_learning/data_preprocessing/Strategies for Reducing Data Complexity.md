![image](https://github.com/user-attachments/assets/372eed1d-3deb-47c2-844d-e0d9f996b0bb)

![image](https://github.com/user-attachments/assets/57bb04cd-e8b2-492f-b76b-76ea5275c5c4)

![image](https://github.com/user-attachments/assets/d620a1ad-75cb-442f-833c-08805a6489b5)

![image](https://github.com/user-attachments/assets/82ce1942-79f3-4891-8e03-00d830a5930e)

![image](https://github.com/user-attachments/assets/0da34c27-9bd0-4ca0-abdb-8617ead906dd)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

# Load data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Initialize PCA
pca = PCA(n_components=2)  # Reduce to two dimensions

# Apply PCA
principalComponents = pca.fit_transform(df)

# Create a DataFrame with the principal components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

print(principalDf.head())
```
