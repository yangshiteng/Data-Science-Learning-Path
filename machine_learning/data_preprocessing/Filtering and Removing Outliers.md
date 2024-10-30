![image](https://github.com/user-attachments/assets/175cd664-ffd3-454a-a876-3eb76f098f15)

![image](https://github.com/user-attachments/assets/9009f32f-10da-4af4-ab68-118fea4710cf)

![image](https://github.com/user-attachments/assets/23474fca-1589-4ad4-bc0f-eb9ef0736151)

```python
import pandas as pd

# Sample data
data = {
    'Feature': [12, 15, -1, 1000, 14, 16, 18, 13, 12, 11, 10, 15]
}
df = pd.DataFrame(data)

# Calculating IQR
Q1 = df['Feature'].quantile(0.25)
Q3 = df['Feature'].quantile(0.75)
IQR = Q3 - Q1

# Defining outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
filtered_df = df[(df['Feature'] >= lower_bound) & (df['Feature'] <= upper_bound)]

print("Original Data:")
print(df)
print("\nData without Outliers:")
print(filtered_df)
```
