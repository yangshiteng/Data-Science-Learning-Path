# Tutorial (all code are under python 3.10)

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

## 5. path parameter (defined in @app.get("/items/{item_id}")) (can not be optional or have default value)

### 5.1 regular path parameter

    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d1b219e7-b673-4a18-a7aa-5d91d097c43c)

### 5.2 path parameter with enumeration (pre-defined value in drop down style)

    from enum import Enum
    
    from fastapi import FastAPI
    
    class ModelName(str, Enum):
        alexnet = "alexnet"
        resnet = "resnet"
        lenet = "lenet"
    
    app = FastAPI()
    
    @app.get("/models/{model_name}")
    def get_model(model_name: ModelName):
        if model_name is ModelName.alexnet:
            return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
        if model_name.value == "lenet":
            return {"model_name": model_name, "message": "LeCNN all the images"}
    
        return {"model_name": model_name, "message": "Have some residuals"}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7d91bc98-36b3-47aa-acda-28fe4e1b1e24)

## 6. query parameter (not defined in @app.get("/items/")) (can be optional and can have default values)

### 6.1 query parameter - default value

    from fastapi import FastAPI
    
    app = FastAPI()
    
    fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
    
    
    @app.get("/items/")
    def read_item(skip: int = 0, limit: int = 10):
        return fake_items_db[skip : skip + limit]

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/85d2d54d-fa31-4fd9-b11a-67efc4e372be)

### 6.2 query parameter - optional value

    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/items/{item_id}")
    def read_item(item_id: str, q: str | None = None):
        if q:
            return {"item_id": item_id, "q": q}
        return {"item_id": item_id}
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f05d25b9-98b5-4522-8bec-976a6b60ea22)

### 6.3 path and query parameters together 

    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/users/{user_id}/items/{item_id}")
    def read_user_item(
        user_id: int, item_id: str, q: str | None = None, short: bool = False
    ):
        item = {"item_id": item_id, "owner_id": user_id}
        if q:
            item.update({"q": q})
        if not short:
            item.update(
                {"description": "This is an amazing item that has a long description"}
            )
        return item

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/66daddc9-b3a2-4b35-abe2-320adc0f0753)

# 7. Request Body and Response Body

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e0dade89-2207-4991-a387-1f8853531cb4)

## 7.1 Request Body

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
    
    
    app = FastAPI()
    
    
    @app.post("/items/")
    def create_item(item: Item):
        item_dict = item.dict()
        if item.tax:
            price_with_tax = item.price + item.tax
            item_dict.update({"price_with_tax": price_with_tax})
        return item_dict


![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/368b5e5c-7cab-43dc-b1bc-e9892c2259f9)






    
