* Pinciple Component Analysis is a dimensionality reduction method
* Each Principle Component is a linear combination of all the variables in a dataset
* Pinciple Components are independent with each other

# Introduction 1 (from Chatgpt)

![image](https://github.com/user-attachments/assets/2f05c764-4147-4802-9519-fa38114815e6)

![image](https://github.com/user-attachments/assets/0c939012-5c20-4e80-bf56-699528122d5b)

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example data
np.random.seed(0)
X_original = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_original[:, 0], X_original[:, 1], alpha=0.2)
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
plt.title('PCA Transformed Data')
plt.show()

# Print the variance explained by each principal component
print("Variance explained by each component:", pca.explained_variance_ratio_)
```

![image](https://github.com/user-attachments/assets/b2166bbf-85d1-4b2d-a319-bc82a3106029)


# Introduction 2 (from Blog) 

![image](https://user-images.githubusercontent.com/60442877/188251602-1dd2eee2-bc73-4f1a-b980-45ab8265141b.png)

![image](https://user-images.githubusercontent.com/60442877/188251607-4024ee2f-cead-49e4-bc99-08a76d9cb152.png)

![image](https://user-images.githubusercontent.com/60442877/188251708-1109e7d6-4fd2-4d13-ad5a-4a0a00e13e7d.png)

![image](https://user-images.githubusercontent.com/60442877/188251739-e34d3ef3-7bf9-4d49-9f54-7d535b3040bc.png)

![image](https://user-images.githubusercontent.com/60442877/188251805-5f1d86fd-6f5c-4915-8117-9b187c2369fc.png)

![image](https://user-images.githubusercontent.com/60442877/188251961-9a107293-ed8a-4af8-93c8-74ad37f794a6.png)

![image](https://user-images.githubusercontent.com/60442877/188252070-9f55b97d-f757-4472-b94c-13bf35f6523b.png)

![image](https://user-images.githubusercontent.com/60442877/188252074-4bbe700b-392b-4f57-8de8-df52194d12d1.png)

![image](https://user-images.githubusercontent.com/60442877/188252077-32546d78-2890-47d3-b687-e1c2ed789982.png)

![image](https://user-images.githubusercontent.com/60442877/188252114-df58941a-d0ee-48f3-a5ed-096c6d5062a9.png)

![image](https://github.com/user-attachments/assets/78e201b9-560a-42af-8959-4222310efff0)

![image](https://github.com/user-attachments/assets/77fa90c4-b0d5-4790-ad81-3f15b44a2b58)
