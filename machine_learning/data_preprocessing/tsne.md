![image](https://github.com/user-attachments/assets/eaaab7b0-aa30-44cb-8a29-092f7fe01887)

![image](https://github.com/user-attachments/assets/8857574f-a0eb-4a4a-bff0-fe3f41ad72ad)

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Simulate data: 10 individuals with varying degrees of relationships
np.random.seed(0)
X = np.random.rand(30, 5)  # 10 individuals, relationships in 5 dimensions

# Applying t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_embedded = tsne.fit_transform(X)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
for i in range(X.shape[0]):
    plt.annotate(f'Person {i+1}', (X_embedded[i, 0], X_embedded[i, 1]))
plt.title('t-SNE visualization of Social Relationships')
plt.show()
```
![image](https://github.com/user-attachments/assets/2bf9ad58-aa5b-49c4-b8c1-56e54736010f)
