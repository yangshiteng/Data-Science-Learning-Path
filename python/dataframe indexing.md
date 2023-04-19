# 1. Basic indexing operator

## Select columns with []
### will raise error if column name not exist

    df['A']

    df[['A','B']]

## Select columns with get()
### will not raise error if column name not exist

![image](https://user-images.githubusercontent.com/60442877/232361276-135d697f-811f-461c-a76e-0f364e989dd8.png)

## Select rows with [] and slicing
### only works for slicing

    df[0:2] (not include row index 2)
    
    df[a:d] (include row index d)

# 2. Label-based indexing - loc[]

## Select a single row

    df.loc[1]
    
    df.loc['b']
    
## Select multiple rows

    df.loc[0:3] (not include row index 3)
    
    df.loc['a':'c'] (include row index c)
    
    df.loc[['a', 'c']]
  
    df.loc[[0,2]]
    
## Select a single value

    df.loc[0, 'A']

    df.loc['b', 'A']
    
## Select a range of rows and columns

    df.loc[0:3, 'A':'C']
    
    df.loc['a':'b', 'A':'C']

# 3. Integer-based indexing - iloc[]

## Select a single row

    df.iloc[0]
    
## Select multiple rows

    df.iloc[0:3]
    
    df.iloc[[0,2]]
    
## Select a single value

    df.loc[0, 1]
    
## Select a range of rows and columns

    df.loc[0:3, 0:2]

# 4. Fast scalar accessing

![image](https://user-images.githubusercontent.com/60442877/232183868-e3678803-6517-4eeb-8017-05b396164396.png)
    
## .at[] - Faster label based indexer for accessing a scalar value (single element) in a DataFrame

    df.at[1, 'A']

    df.at['a', 'A']
    
## .iat[] - Faster integer based indexer for accessing a scalar value (single element) in a DataFrame

    df.iat[1,0]


# 5. Boolean indexing

## .query() - Method for querying a DataFrame using boolean expressions

    # Selecting rows where the value in column 'A' is greater than 1
    subset_df = df.query("A > 1")
    print(subset_df)

## .mask() - Method to mask elements in a DataFrame based on a condition

    # Mask values greater than 4
    masked_df = df.mask(df > 4)
    print(masked_df)

## Filter data based on a boolean condition 

    df[(df['A'] > 2) & (df['B'] < 9)]

# 6. filter()

![image](https://user-images.githubusercontent.com/60442877/232359430-5536ee2f-fd1f-40eb-9466-3a6754e6a75b.png)

![image](https://user-images.githubusercontent.com/60442877/232360255-316203ee-0781-4953-b9cf-0d90d1ded429.png)

# 7. truncate()

![image](https://user-images.githubusercontent.com/60442877/232947955-3a64f797-da82-4d38-8e8f-45a1b178a8fd.png)

![image](https://user-images.githubusercontent.com/60442877/232947972-9842cbc8-8267-4e81-bb37-edcf0a777105.png)

![image](https://user-images.githubusercontent.com/60442877/232947993-7de966c9-e6dd-4713-84a8-d45a6d6bddfc.png)



