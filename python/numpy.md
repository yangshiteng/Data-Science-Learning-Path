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
    











