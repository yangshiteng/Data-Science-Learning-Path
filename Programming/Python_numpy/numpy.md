# 1. Numpy Introduction

![image](https://user-images.githubusercontent.com/60442877/229387816-3f379ad2-f253-4463-8bde-cb1007643ecc.png)

![image](https://user-images.githubusercontent.com/60442877/229389457-9d00fc45-c3c4-4537-8594-258b021a9d98.png)

# 2. Create a numpy ndarray object

![image](https://user-images.githubusercontent.com/60442877/229389627-84ed8fc2-23fb-40a7-b844-886ab59183c9.png)

![image](https://user-images.githubusercontent.com/60442877/229389635-633b7123-e4ff-4110-8589-a12f36150644.png)

![image](https://user-images.githubusercontent.com/60442877/229389678-f49733c3-8622-4245-a02a-b5632aaba859.png)

# 3. Dimensions in numpy ndarray

![image](https://user-images.githubusercontent.com/60442877/229389800-1501aabb-c561-47c7-a75b-afe208f85e59.png)

## 0-D Array (just scalar)

![image](https://user-images.githubusercontent.com/60442877/229389886-8580dd75-1d25-4ede-9d1d-4abada6e7f02.png)

![image](https://user-images.githubusercontent.com/60442877/229389894-551b051c-d964-46b4-adc9-a4168c642b36.png)

## 1-D Array 

![image](https://user-images.githubusercontent.com/60442877/229390237-b7ca68b3-bc97-40e0-82be-5382ee84147f.png)

![image](https://user-images.githubusercontent.com/60442877/229390245-ca124a52-5f26-4bad-8278-c83186db3964.png)

## 2-D Array

![image](https://user-images.githubusercontent.com/60442877/229390357-8ed69482-c57b-45a2-87e4-77077c595123.png)

![image](https://user-images.githubusercontent.com/60442877/229390365-05cc10aa-b4c8-47f5-9a7b-7d674d184eb8.png)

## 3-D Array

![image](https://user-images.githubusercontent.com/60442877/229390434-899d6dfb-54ec-4695-9265-425982873918.png)

![image](https://user-images.githubusercontent.com/60442877/229390441-5e81a920-b245-4886-b591-ce493293d87c.png)

# 4. Check Number of Dimensions

![image](https://user-images.githubusercontent.com/60442877/229390611-569a8d85-2f1c-4f74-bf65-aaae2e2ee1ed.png)

![image](https://user-images.githubusercontent.com/60442877/229390613-e195bd2f-fd70-4dc5-99cd-d30bb00ebcb8.png)

# 5. Create a numpy ndarray with specified number of dimensions

![image](https://user-images.githubusercontent.com/60442877/229390786-c740aea4-d34b-4033-8459-cca47e58599c.png)

![image](https://user-images.githubusercontent.com/60442877/229390796-9bb5216b-01fa-4fc6-a357-611b7fc00e3c.png)

# 6. Numpy Array Indexing

## Access 1-D Array

![image](https://user-images.githubusercontent.com/60442877/229390983-0be1e880-f00f-44e2-9ca5-9e343105020d.png)

    import numpy as np

    arr = np.array([1, 2, 3, 4])

    print(arr[0])
    # return 1

## Access 2-D Array

![image](https://user-images.githubusercontent.com/60442877/229393544-52836ced-bf61-499a-8374-2b33c7019376.png)

    import numpy as np

    arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

    print('2nd element on 1st row: ', arr[0, 1])
    # return 2nd element on 1st dim:  2

## Access 3-D Array

    import numpy as np

    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    print(arr[0, 1, 2])
    # return 6
    
## Negative Indexing

![image](https://user-images.githubusercontent.com/60442877/229393690-be648702-7559-41fc-9bf1-d17321b56a54.png)

![image](https://user-images.githubusercontent.com/60442877/229393696-aa9d4e5d-d3fa-4560-bd27-532a92374fd1.png)

# 7. Numpy Array Slicing

![image](https://user-images.githubusercontent.com/60442877/229394389-9b0f638a-b333-4e09-8308-0531cfb5200b.png)

![image](https://user-images.githubusercontent.com/60442877/229395308-881c6766-e91e-4c89-ab59-b9aefd89c203.png)

## 1-D Array Slicing

    import numpy as np

    arr = np.array([1, 2, 3, 4, 5, 6, 7])

    print(arr[1:5])
    # return [2 3 4 5]

    print(arr[4:])
    # return [5 6 7]
    
    print(arr[:4])
    # return [1 2 3 4]
    
    print(arr[-3:-1])
    # return [5 6]
    
    print(arr[1:5:2])
    # return [2 4]
    
    print(arr[::2])
    # return [1 3 5 7]
    
## 2-D Array Slicing

    import numpy as np

    arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    print(arr[1, 1:4])
    # return [7 8 9]

    print(arr[0:2, 2])
    # return [3 8]
    
    print(arr[0:2, 1:4])
    # return [[2 3 4]
              [7 8 9]]
    
# 8. Data Types in Numpy

![image](https://user-images.githubusercontent.com/60442877/229395411-1ec456ab-9e27-4426-882f-5fee39fcaea0.png)

## Data Type Checking 

    import numpy as np

    arr = np.array([1, 2, 3, 4])

    print(arr.dtype)
    # return int64

## Create a numpy ndarray with a defined Data Type

![image](https://user-images.githubusercontent.com/60442877/229395744-e91de665-252e-49ba-8298-dc50c90f9b3b.png)

![image](https://user-images.githubusercontent.com/60442877/229395757-d0d7d5a4-55dc-43e9-8466-19a392b87576.png)

![image](https://user-images.githubusercontent.com/60442877/229395812-5c9f3c3b-e9d0-4aa3-88af-e6f291b65cd2.png)

## Data Type Convert

    import numpy as np

    arr = np.array([1.1, 2.1, 3.1])

    newarr = arr.astype('i') 
    # newarr = arr.astype(int)

    print(newarr)       # [1 2 3]
    print(newarr.dtype) # int32
    
# 9. Copy vs View

![image](https://user-images.githubusercontent.com/60442877/229397171-7d981539-d40e-432b-9092-da5f327c251d.png)

    # Copy Example
    
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])
    x = arr.copy()
    arr[0] = 42

    print(arr) # [42  2  3  4  5]
    print(x)   # [1 2 3 4 5] 

    # View Example

    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])
    x = arr.view()
    arr[0] = 42

    print(arr)  # [42  2  3  4  5]
    print(x)    # [42  2  3  4  5]

## Check if Array owns its data

![image](https://user-images.githubusercontent.com/60442877/229397496-6e1be3ca-00e2-4401-91a1-160bd78acb70.png)

    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])

    x = arr.copy()
    y = arr.view()

    print(x.base) # None
    print(y.base) # [1 2 3 4 5] 

# 10. Shape of a numpy ndarray

    import numpy as np

    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    print(arr.shape) 
    # (2, 4)

    arr = np.array([1, 2, 3, 4], ndmin=5)

    print(arr)
    # [[[[[1 2 3 4]]]]]
    
    print('shape of array :', arr.shape)
    # shape of array : (1, 1, 1, 1, 4)

# 11. Numpy Array Reshaping

![image](https://user-images.githubusercontent.com/60442877/229398836-5e55492b-21da-4fff-ace1-83359f78b25a.png)

## Reshape from 1-D to 2-D

![image](https://user-images.githubusercontent.com/60442877/229402387-4265d480-58da-47d7-abe5-4cff06860b3c.png)

![image](https://user-images.githubusercontent.com/60442877/229402395-d5c7a7f8-5235-49d1-a4de-866b443601cf.png)

## Reshape from 1-D to 3-D

![image](https://user-images.githubusercontent.com/60442877/229402448-afb2f078-c6db-4f54-bc3a-900468a2fd8a.png)

![image](https://user-images.githubusercontent.com/60442877/229402457-c2aa6d20-155f-45cf-a9c3-d7a92ddc5da6.png)

## Can we reshape into any shape?

![image](https://user-images.githubusercontent.com/60442877/229402566-bb40c4c3-5699-4a57-ad36-d02a9f075829.png)

## Unknow Dimension

![image](https://user-images.githubusercontent.com/60442877/229402644-90de49ac-c484-4bad-aab1-1daf60245635.png)

![image](https://user-images.githubusercontent.com/60442877/229402655-52abc54c-5a10-4310-8900-a0eb2547f4f8.png)

## Flattening the Array

![image](https://user-images.githubusercontent.com/60442877/229402739-93e7f5ab-9a74-437a-957c-d3b95b150933.png)

![image](https://user-images.githubusercontent.com/60442877/229402757-ae87df0e-be04-4eb4-801d-c1cf2ff1aeba.png)

# 12. Numpy Array Iterating

## 1-D Array Iterating

    import numpy as np

    arr = np.array([1, 2, 3])

    for x in arr:
      print(x)
      # 1
      # 2
      # 3

## 2-D Array Iterating

    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
      print(x)
      # [1 2 3]
      # [4 5 6]

![image](https://user-images.githubusercontent.com/60442877/229957442-80f4bcd5-9673-448e-909f-de057eb9918d.png)

    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
      for y in x:
        print(y)
        # 1
        # 2
        # 3
        # 4
        # 5
        # 6
        
## 3-D Array Iterating

![image](https://user-images.githubusercontent.com/60442877/229957774-22d700c9-11f7-41d6-b3f5-9454a3328d73.png)

![image](https://user-images.githubusercontent.com/60442877/229957783-c0af0d74-a8fb-45af-8044-69927925d74f.png)

![image](https://user-images.githubusercontent.com/60442877/229957835-edbaaf04-6081-4454-a486-f40dde8f2e0a.png)

![image](https://user-images.githubusercontent.com/60442877/229957843-f887326d-f375-44a1-98df-3107c83e1e5d.png)

## Iterating Arrays Using nditer()

![image](https://user-images.githubusercontent.com/60442877/229958078-2b3a769b-c369-4c64-9200-2d6dc67946e3.png)

![image](https://user-images.githubusercontent.com/60442877/229958095-3e0b27cf-1933-4178-8f53-9afc32c95c1b.png)

![image](https://user-images.githubusercontent.com/60442877/229958361-4206b7b8-dba9-4bd4-9c34-8c6e4634bb82.png)

![image](https://user-images.githubusercontent.com/60442877/229958382-e2aec79b-741e-49ed-9137-16166f0c90cb.png)

## Enumerated Iteration Using ndenumerate()

![image](https://user-images.githubusercontent.com/60442877/229958495-ebe41fa2-77c8-4ecf-a824-1d9742c9d7ec.png)

![image](https://user-images.githubusercontent.com/60442877/229958505-a82e152e-d1cc-4d0f-a3fc-c585066eb481.png)

![image](https://user-images.githubusercontent.com/60442877/229958521-6e8596fa-84a7-4065-8692-007660fb6660.png)

![image](https://user-images.githubusercontent.com/60442877/229958531-8d215a7f-8562-4036-b0d7-d758f1368e89.png)

# 13.NumPy Joining Array

## np.concatenate()

    import numpy as np

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.concatenate((arr1, arr2))
    print(arr)
    # [1 2 3 4 5 6]


    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    arr = np.concatenate((arr1, arr2), axis=1)
    print(arr)
    # [[1 2 5 6]
       [3 4 7 8]]
       
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    arr = np.concatenate((arr1, arr2), axis=1)
    print(arr)
    # [[1 2]
       [3 4]
       [5 6]
       [7 8]]

## np.stack()

    import numpy as np

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.stack((arr1, arr2), axis=1)
    print(arr)
    # [[1 4]
    #  [2 5]
    #  [3 6]]

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.stack((arr1, arr2), axis=0)
    print(arr)
    # [[1 2 3]
       [4 5 6]]
   
## np.hstack()


    import numpy as np

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.hstack((arr1, arr2))
    print(arr)
    # [1 2 3 4 5 6]

## np.vstack()

    import numpy as np
    
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.vstack((arr1, arr2))
    print(arr)
    # [[1 2 3]
       [4 5 6]]
       
## np.dstack()

    import numpy as np

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr = np.dstack((arr1, arr2))
    print(arr)
    # [[[1 4]
        [2 5]
        [3 6]]]


# 14. NumPy Array Split - np.array_split()

![image](https://user-images.githubusercontent.com/60442877/229964669-fbcc20c5-4c91-49ac-b5e5-a9217c07aa5e.png)

![image](https://user-images.githubusercontent.com/60442877/229964740-aa79754d-de19-4440-a45e-49ecad8cc30c.png)

![image](https://user-images.githubusercontent.com/60442877/229964776-28d7c6e5-8832-4a17-a404-c51b1f18b385.png)

![image](https://user-images.githubusercontent.com/60442877/229964805-431c6074-5d4f-41dd-9bf4-2330807272d9.png)

# 15. NumPy Array Search - np.where()

![image](https://user-images.githubusercontent.com/60442877/229966728-0ee8a16a-9535-44ea-bec9-d20679d07600.png)

![image](https://user-images.githubusercontent.com/60442877/229966765-31dfbd47-29b1-43e5-956f-5f880553a0d8.png)

![image](https://user-images.githubusercontent.com/60442877/229966795-5072960f-6244-4d3d-96c5-947be529e331.png)

![image](https://user-images.githubusercontent.com/60442877/229966810-8113f0ae-4510-4c0f-ba7d-a4f914bb63d0.png)

# 16. Numpy ufuncs (Vectorization)

## Introduction

![image](https://user-images.githubusercontent.com/60442877/229967610-19fbfc83-ddd5-4fca-84cd-ef7443881110.png)

![image](https://user-images.githubusercontent.com/60442877/229967635-d3e79873-4f75-4ddb-973c-e618beb21209.png)

![image](https://user-images.githubusercontent.com/60442877/229967658-d08b2530-3ed2-46d8-9ef3-b22f09011147.png)

![image](https://user-images.githubusercontent.com/60442877/229967684-43c10f75-edf1-4236-8df3-af5539e302e8.png)

## Create Your Own ufunc

![image](https://user-images.githubusercontent.com/60442877/229968118-dcdf66e4-965b-492c-b37c-49cb33c39ead.png)

![image](https://user-images.githubusercontent.com/60442877/229968132-bfb009e5-af42-4f60-8018-52c58473cbc3.png)

![image](https://user-images.githubusercontent.com/60442877/229968153-57fafe4f-9470-460f-ac76-c82ee926bf5a.png)

# 17. NumPy Sort - np.sort()

![image](https://user-images.githubusercontent.com/60442877/229972193-22e82717-8582-4b77-889a-92880b9459aa.png)

![image](https://user-images.githubusercontent.com/60442877/229972221-ecab5c0b-8c06-4939-bd46-af53d40d6879.png)

![image](https://user-images.githubusercontent.com/60442877/229972249-e4b85097-2053-40d3-8a5b-b29b44f58bd7.png)


# 18. Numpy Array Filter

![image](https://user-images.githubusercontent.com/60442877/229973101-f491ba25-ae72-48d1-8684-48b282746434.png)

![image](https://user-images.githubusercontent.com/60442877/229973116-c7199cb7-fc38-4e59-9cd4-0d63cf628ff4.png)

![image](https://user-images.githubusercontent.com/60442877/229973143-f9672316-eb74-4a65-a218-bc9d2751dd78.png)

![image](https://user-images.githubusercontent.com/60442877/229973177-6119231f-098b-4ae7-be3c-4dbb4acb60c2.png)


# 19. NumPy Basic Calculation

## np.add()

    import numpy as np

    arr1 = np.array([10, 11, 12, 13, 14, 15])
    arr2 = np.array([20, 21, 22, 23, 24, 25])

    newarr = np.add(arr1, arr2)

    print(newarr)
    # [30 32 34 36 38 40]
    
## np.subtract()

    import numpy as np

    arr1 = np.array([10, 20, 30, 40, 50, 60])
    arr2 = np.array([20, 21, 22, 23, 24, 25])

    newarr = np.subtract(arr1, arr2)

    print(newarr)
    # [-10  -1   8  17  26  35]

## np.multiply()

    import numpy as np

    arr1 = np.array([10, 20, 30, 40, 50, 60])
    arr2 = np.array([20, 21, 22, 23, 24, 25])

    newarr = np.multiply(arr1, arr2)

    print(newarr)
    # [ 200  420  660  920 1200 1500]


## np.divide()

    import numpy as np

    arr1 = np.array([10, 20, 30, 40, 50, 60])
    arr2 = np.array([3, 5, 10, 8, 2, 33])

    newarr = np.divide(arr1, arr2)

    print(newarr)
    # [ 3.33333333  4.          3.          5.         25.          1.81818182]


## np.power()

    import numpy as np

    arr1 = np.array([10, 20, 30, 40, 50, 60])
    arr2 = np.array([3, 5, 6, 8, 2, 33])

    newarr = np.power(arr1, arr2)

    print(newarr)
    # [         1000       3200000     729000000 6553600000000          2500             0]


## np.mod() or np.remainder()

    import numpy as np

    arr1 = np.array([10, 20, 30, 40, 50, 60])
    arr2 = np.array([3, 7, 9, 8, 2, 33])

    newarr = np.mod(arr1, arr2)
    newarr = np.remainder(arr1, arr2)
    
    print(newarr)
    # [ 1  6  3  0  0 27]

## np.absolute()

    import numpy as np

    arr = np.array([-1, -2, 1, 2, 3, -4])

    newarr = np.absolute(arr)

    print(newarr)
    # 1 2 1 2 3 4]

## np.trunc() or np.fix()

    import numpy as np

    arr = np.trunc([-3.1666, 3.6667])
    arr = np.fix([-3.1666, 3.6667])
    
    print(arr)
    # [-3.  3.]

## np.around()

![image](https://user-images.githubusercontent.com/60442877/229975706-f76a95ed-0ab8-4342-8d1e-7faa5d489b2f.png)

    import numpy as np

    arr = np.around(3.1666, 2)

    print(arr)
    # 3.17

## np.floor()

    import numpy as np

    arr = np.floor([-3.1666, 3.6667])

    print(arr)
    # [-4.  3.]
    
## np.ceil()

    import numpy as np

    arr = np.ceil([-3.1666, 3.6667])

    print(arr)
    # [-3.  4.]
    
## np.log2() - log at the base 2

## np.log10() - log at the base 10

## np.log() - log at the base e

## log at any base

![image](https://user-images.githubusercontent.com/60442877/229976347-804be640-8fc6-4d43-8569-37e6469bc88a.png)

# 20. Numpy Summations

## Difference between summation and addition

![image](https://user-images.githubusercontent.com/60442877/230698043-913e8417-fb54-4026-8d00-e25244b8ba00.png)

![image](https://user-images.githubusercontent.com/60442877/230698046-31a2e5a8-00c3-4591-9b72-0efef97b99c9.png)

## Summation in 3 ways

    import numpy as np

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([1, 2, 3])
    
    # first way
    newarr = np.sum([arr1, arr2, arr3])
    print(newarr)
    # 18

    # second way
    newarr = np.sum([arr1, arr2, arr3], axis=0)
    print(newarr)
    # [3 6 9]

    # third way
    newarr = np.sum([arr1, arr2, arr3], axis=1)
    print(newarr)
    # [6 6 6]

## Cumulative Sum - np.cumsum()

![image](https://user-images.githubusercontent.com/60442877/230697939-d08dab45-fb3d-4e51-8b84-26b2cad094ca.png)

![image](https://user-images.githubusercontent.com/60442877/230697944-658c580a-b87b-42ad-8d6d-97e178cdcf12.png)

# 21. Numpy Products

## np.prod()

![image](https://user-images.githubusercontent.com/60442877/230698055-68475b86-bd46-4809-b0e5-a90f84e20d60.png)

![image](https://user-images.githubusercontent.com/60442877/230698083-c5ffc496-6a6b-47cb-b321-716ad8b7c249.png)

## Product Over an Axis

    import numpy as np

    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([5, 6, 7, 8])

    newarr = np.prod([arr1, arr2], axis=1)
    print(newarr)
    # [24 1680]

    newarr = np.prod([arr1, arr2], axis=0)
    print(newarr)
    # [5 12 21 32]

## Cummulative Product - np.cumprod()

![image](https://user-images.githubusercontent.com/60442877/230698130-d4fa3736-5836-4502-bfcc-88f29413362e.png)

# 21. Numpy Difference - np.diff()

![image](https://user-images.githubusercontent.com/60442877/230814996-6e0e2eb3-b323-42a4-9ec2-e49c64cb48fb.png)

![image](https://user-images.githubusercontent.com/60442877/230815087-45367a9e-b016-411d-b865-59044c284121.png)

# 22. LCM (Lowest Common Multiple) and GCD

## LCM

![image](https://user-images.githubusercontent.com/60442877/230815203-53dd7fb4-0518-4a5f-94a2-e568927b4438.png)

![image](https://user-images.githubusercontent.com/60442877/230815382-080c8806-ddf0-4108-8fbb-3ee5a2d06f22.png)

    import numpy as np

    arr = np.arange(1, 11)

    x = np.lcm.reduce(arr)

    print(x)
    # 2520

## GCD

![image](https://user-images.githubusercontent.com/60442877/230815554-f9781473-f908-43fd-aba0-a6d071cbc6f9.png)

![image](https://user-images.githubusercontent.com/60442877/230815697-9362f0de-86da-4986-a47c-cd95afc0c4dc.png)

# 23. Numpy Set Operations (union, intersection, difference, symmetric difference)

![image](https://user-images.githubusercontent.com/60442877/230817039-7129ea85-f3d5-4088-a45e-c636b89fc99a.png)

    import numpy as np

    arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    x = np.unique(arr)

    print(x)
    # [1 2 3 4 5 6 7]
    
## Finding Union - np.union1d()

    import numpy as np

    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([3, 4, 5, 6])

    newarr = np.union1d(arr1, arr2)

    print(newarr)
    # [1 2 3 4 5 6]

## Finding Intersection - np.intersection1d()

    import numpy as np

    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([3, 4, 5, 6])

    newarr = np.intersect1d(arr1, arr2, assume_unique=True)

    print(newarr)
    # [3 4]

![image](https://user-images.githubusercontent.com/60442877/230817417-c04b2325-219f-422c-b676-3f2373a1ba68.png)

## Finding Difference - np.setdiff1d()

### To find only the values in the first set that is NOT present in the seconds set, use the setdiff1d() method.

    import numpy as np

    set1 = np.array([1, 2, 3, 4])
    set2 = np.array([3, 4, 5, 6])

    newarr = np.setdiff1d(set1, set2, assume_unique=True)

    print(newarr)
    # [1 2]

## Finding Symmetric Difference - np.setxor1d()

    import numpy as np

    set1 = np.array([1, 2, 3, 4])
    set2 = np.array([3, 4, 5, 6])

    newarr = np.setxor1d(set1, set2, assume_unique=True)

    print(newarr)
    # [1 2 5 6]


# 22. np.arange()

![image](https://user-images.githubusercontent.com/60442877/230819140-cdd8df07-358d-4488-a843-a9979fdd4b01.png)

![image](https://user-images.githubusercontent.com/60442877/230819177-82411694-f33b-47de-ad41-b32dd04a29ca.png)

![image](https://user-images.githubusercontent.com/60442877/230819200-103d2b92-703b-4c81-b504-dfe3432fafae.png)



    
