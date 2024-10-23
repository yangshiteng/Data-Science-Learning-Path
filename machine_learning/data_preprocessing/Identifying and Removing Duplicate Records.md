![image](https://github.com/user-attachments/assets/feab74c9-3268-4749-bdf4-e4d456f75bcd)

![image](https://github.com/user-attachments/assets/5b71ed69-07be-4fff-b25f-adaf53bdb1d2)

![image](https://github.com/user-attachments/assets/f5187402-1c2c-41d2-84df-b2cb9c612748)

![image](https://github.com/user-attachments/assets/43314b35-8ce1-41c2-a2af-b7a96a951f91)

# Pytnon Examples

## Example: Removing Exact Duplicates with Pandas

```pytnon
import pandas as pd

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Bob', 'Charlie', 'Charlie'],
    'Age': [25, 30, 35, 30, 35, 35],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Los Angeles', 'Chicago', 'Chicago']
}

df = pd.DataFrame(data)

# Displaying original DataFrame
print("Original DataFrame:")
print(df)

# Removing duplicates: default keeps the first occurrence
df_unique = df.drop_duplicates()

# Displaying DataFrame after removing duplicates
print("\nDataFrame After Removing Duplicates:")
print(df_unique)
```

## Example: Removing Near Duplicates with Fuzzywuzzy

```python
from fuzzywuzzy import process
import pandas as pd

# Sample data
names = ['Alice', 'Alicia', 'Alias', 'Bob', 'Bobby', 'Charlie', 'Charlee']

# Converting list to DataFrame
df_names = pd.DataFrame(names, columns=['Name'])

# Threshold for considering as duplicate
similarity_threshold = 80  # percentage

# Function to find duplicates based on fuzzy matching
def find_fuzzy_duplicates(df, column, threshold=90):
    duplicates = []
    # Comparing each item with every other item
    for i in range(len(df)):
        name = df.iloc[i][column]
        # Get matches above the threshold and not the exact same string
        matches = process.extract(name, df[column], limit=None, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
        # Filter matches
        for match in matches:
            if match[1] >= threshold and match[0] != name:
                duplicates.append((name, match[0], match[1]))
    return duplicates

# Finding duplicates
duplicates = find_fuzzy_duplicates(df_names, 'Name', similarity_threshold)

# Display results
print("Potential Duplicates:")
for dup in duplicates:
    print(f"{dup[0]} is similar to {dup[1]} with similarity {dup[2]}%")

# Deciding on duplicates handling is subjective and may require manual intervention
```







