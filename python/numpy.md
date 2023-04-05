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




