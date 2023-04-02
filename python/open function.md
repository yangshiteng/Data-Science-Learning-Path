#  Introduction

![image](https://user-images.githubusercontent.com/60442877/229324582-c78aa4c0-93e6-47ad-ae31-e4668d987f4e.png)

# "r" mode - default value, open a file for reading, error if the file doe not exist

## 1. Open a file on the server

![image](https://user-images.githubusercontent.com/60442877/229325152-970562e2-b9c8-4406-897c-fc31c6ff457f.png)

![image](https://user-images.githubusercontent.com/60442877/229325214-b361ff00-c35b-4a7a-9a8e-cdffa7ffc7dc.png)

![image](https://user-images.githubusercontent.com/60442877/229325172-5836b753-5c75-434b-911a-9092e2e3db21.png)

![image](https://user-images.githubusercontent.com/60442877/229325216-7c2df488-afed-4557-8cb1-9c72333dbe0f.png)

## 2. Read only parts of the file

![image](https://user-images.githubusercontent.com/60442877/229325187-0cf0c1a4-1207-4f4e-8932-d1e27f2b934c.png)

![image](https://user-images.githubusercontent.com/60442877/229325197-3717dc10-97a7-4988-a0c0-0c2d6573ca35.png)

## 3. Read Lines

![image](https://user-images.githubusercontent.com/60442877/229325259-2d3082bf-a732-403a-8567-53da46cf8f43.png)

![image](https://user-images.githubusercontent.com/60442877/229325295-18d0c91e-153b-4f74-8325-d5b97a1ec91e.png)

![image](https://user-images.githubusercontent.com/60442877/229325301-2ddf3092-8216-43a4-81fc-eff096c84547.png)

![image](https://user-images.githubusercontent.com/60442877/229325305-56c301f1-1044-424b-968f-0f7cb3cf9b6f.png)

## 4. Looping the lines of the file

![image](https://user-images.githubusercontent.com/60442877/229325363-dd06348f-bfdc-4d1b-8edf-38e576826ccc.png)

![image](https://user-images.githubusercontent.com/60442877/229325367-75e5b5d2-a3bc-4d3a-8d7d-7ce8d4f8eb38.png)

## 5. Close Files

![image](https://user-images.githubusercontent.com/60442877/229326459-354554cb-4344-49af-bbdf-64a3de39a7a9.png)

![image](https://user-images.githubusercontent.com/60442877/229326468-abeed1ed-f73f-429c-8270-41ec75ce97c8.png)

# "a" mode - open a file for appending, create the file if it does not exist

## 1. Append the file

![image](https://user-images.githubusercontent.com/60442877/229327080-1c9ab5cb-8157-48b2-bc6b-7f29c1185040.png)

![image](https://user-images.githubusercontent.com/60442877/229327082-c9577f1d-47dc-4f5d-9f25-b4d5b36edd85.png)

## 2. Create the file if the specified file does not exist

    f = open("new.txt", "a")
    f.write("Now the file is just created")
    f.close()
    # create a new file new.txt since this file does not exist
    # and also write one sentence into this file

    # open and read this file:
    f = open("new.txt", "r")
    print(f.read())
    f.close()
    # output: "Now the file is just created"
    
# "w" mode - open a file for overwriting, create the file if it does not exist

## 1. Overwrite the file

![image](https://user-images.githubusercontent.com/60442877/229327544-4c5e2644-06b2-41bd-ad27-92de3ef65f40.png)

![image](https://user-images.githubusercontent.com/60442877/229327547-db478c83-4eb2-4b28-8a79-b8e39302a23e.png)

## 2. Create the file if it does not exist

    f = open("new.txt", "w")
    f.write("Now the file is just created")
    f.close()
    # create a new file new.txt since this file does not exist
    # and also write one sentence into this file

    # open and read this file:
    f = open("new.txt", "r")
    print(f.read())
    f.close()
    # output: "Now the file is just created"

# "x" mode - create the specified file, return an error if the file exists

![image](https://user-images.githubusercontent.com/60442877/229327608-434771d1-c366-43bf-9f97-4131ee64f2a3.png)

