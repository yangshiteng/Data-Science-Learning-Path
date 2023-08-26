# Tutorial

https://fastapi.tiangolo.com/

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7be593d3-2758-4fcc-943a-d5977006c500)

## 1. Install the fast api

    # install it with all the optional dependencies and features
    # that also includes uvicorn, that you can use as the server that runs your code
    
    pip install "fastapi[all]"

    # you can also install it part by part
    
    pip install fastapi
    pip install "uvicorn[standard]"

## 2. Create a simplest FastAPI file and run the server

    # Copy the following code to a file main.py
    
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"message": "Hello World"}

    # run the following code in your terminal
    
    uvicorn main:app --reload

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a96a60b5-bfa3-47d9-b853-5510c443fadc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0cbf93db-e582-4c83-aad6-2319ddcb3706)

## 3. Pydantic models - data validation

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ec3d5321-4374-447f-9f4a-01d7ce991518)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5a07e9c4-57f0-4c8d-9b82-312acf4a0bf3)

## 4. HTTP methods

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4e877b90-37c0-4af4-b65d-c3699844c16f)

## 5. path parameter

### 5.1 regular path parameter

    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/items/{item_id}")
    async def read_item(item_id: int):
        return {"item_id": item_id}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3eb028f6-555e-4f11-81d9-e8ae82c75d6f)

### 5.2 path parameter with enumeration (pre-defined value in drop down style)

    from enum import Enum
    
    from fastapi import FastAPI
    
    class ModelName(str, Enum):
        alexnet = "alexnet"
        resnet = "resnet"
        lenet = "lenet"
    
    app = FastAPI()
    
    @app.get("/models/{model_name}")
    async def get_model(model_name: ModelName):
        if model_name is ModelName.alexnet:
            return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
        if model_name.value == "lenet":
            return {"model_name": model_name, "message": "LeCNN all the images"}
    
        return {"model_name": model_name, "message": "Have some residuals"}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/679403d8-288b-4dac-96f5-c5267be97b88)











    
