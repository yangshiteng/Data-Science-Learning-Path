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

    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}

## 6. query parameter (not defined in @app.get("/items/")) (can be optional and can have default values)

### 6.1 query parameter - required value

    from fastapi import FastAPI
    
    app = FastAPI()
    
    fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
    
    @app.get("/items/")
    def read_item(skip: int, limit: int):
        return fake_items_db[skip : skip + limit]

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/414c4b2c-5408-4882-830d-fdb3ae99d40b)


### 6.2 query parameter - default value

    from fastapi import FastAPI
    
    app = FastAPI()
    
    fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
    
    
    @app.get("/items/")
    def read_item(skip: int = 0, limit: int = 10):
        return fake_items_db[skip : skip + limit]

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/85d2d54d-fa31-4fd9-b11a-67efc4e372be)

### 6.3 query parameter - optional value

    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/items/{item_id}")
    def read_item(item_id: str, q: str | None = None):
        if q:
            return {"item_id": item_id, "q": q}
        return {"item_id": item_id}
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f05d25b9-98b5-4522-8bec-976a6b60ea22)

### 6.4 path and query parameters together 

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

## 7. Body Parameter (can also set up example request and response)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e0dade89-2207-4991-a387-1f8853531cb4)

### 7.1 Body Parameter - Requst Body

#### 7.1.1 request body

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

#### 7.1.2 request body + path parameter + query parameter

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
    
    app = FastAPI()
    
    @app.put("/items/{item_id}")
    def create_item(item_id: int, item: Item, q: str | None = None):
        result = {"item_id": item_id, **item.dict()}
        if q:
            result.update({"q": q})
        return result

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/11410241-b470-4094-bf9c-8a58b417f7f6)

#### 7.1.3 request body - Field (set up parameter within request body)

    from typing import Annotated
    from fastapi import Body, FastAPI
    from pydantic import BaseModel, Field
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str | None = None
        description: str | None = Field(default=None, max_length=300)
        price: float = Field(gt=0)
        tax: float = Field(default=50, max_length=300)
    
    @app.put("/items/{item_id}")
    async def update_item(item_id: int, item: Annotated[Item, Body(embed=True)]):
        results = {"item_id": item_id, "item": item}
        return results

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cb59b685-c143-4852-88cd-6a6f4e78b485)

#### 7.1.4 request body - list

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
        tags: list[str] = []
    
    @app.put("/items/{item_id}")
    def update_item(item_id: int, item: Item):
        results = {"item_id": item_id, "item": item}
        return results
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4559c566-9cdb-42f6-bfa5-4297ba3a61f2)

#### 7.1.5 request body - nested

    from fastapi import FastAPI
    from pydantic import BaseModel, HttpUrl
    
    app = FastAPI()
    
    class Image(BaseModel):
        url: HttpUrl
        name: str
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
        tags: set[str] = set()
        images: list[Image] | None = None
    
    @app.put("/items/{item_id}")
    def update_item(item_id: int, item: Item):
        results = {"item_id": item_id, "item": item}
        return results

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0804e5c8-77f4-468a-a6d0-a2aad669acb6)

#### 7.1.6 request body - example data

    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str = Field(examples=["Foo"])
        description: str | None = Field(default=None, examples=["A very nice Item"])
        price: float = Field(examples=[35.4])
        tax: float | None = Field(default=None, examples=[3.2])
    
    @app.put("/items/{item_id}")
    async def update_item(item_id: int, item: Item):
        results = {"item_id": item_id, "item": item}
        return results

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ef813cbe-8c62-404e-8eec-e16485822b6d)

### 7.2 Body Parameter - Response Body

#### 7.2.1 response body - Return Type

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
        tags: list[str] = []
    
    @app.post("/items/")
    def create_item(item: Item) -> Item:
        return item
    
    @app.get("/items/")
    def read_items() -> list[Item]:
        return [
            Item(name="Portal Gun", price=42.0),
            Item(name="Plumbus", price=32.0),
        ]

#### 7.2.2 response body - set up with response_model 

    from typing import Any
    
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
        tags: list[str] = []
    
    @app.post("/items/", response_model=Item)
    async def create_item(item: Item) -> Any:
        return item
    
    @app.get("/items/", response_model=list[Item])
    def read_items() -> Any:
        return [
            {"name": "Portal Gun", "price": 42.0},
            {"name": "Plumbus", "price": 32.0},
        ]

#### 7.2.3 response body - exclude or include default values

##### response_model_exclude_unset

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()

    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float = 10.5
        tags: list[str] = []
    
    items = {
        "foo": {"name": "Foo", "price": 50.2},
        "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
        "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
    }
    
    @app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True) 
    # True if you want to exclude default value, False if you want to keep default value
    def read_item(item_id: str):
        return items[item_id]

##### response_model_include & response_model_exclude

    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float = 10.5
    
    items = {
        "foo": {"name": "Foo", "price": 50.2},
        "bar": {"name": "Bar", "description": "The Bar fighters", "price": 62, "tax": 20.2},
        "baz": {
            "name": "Baz",
            "description": "There goes my baz",
            "price": 50.2,
            "tax": 10.5,
        },
    }
    
    @app.get(
        "/items/{item_id}/name",
        response_model=Item,
        response_model_include={"name", "description"},
    )
    async def read_item_name(item_id: str):
        return items[item_id]
    
    @app.get("/items/{item_id}/public", response_model=Item, response_model_exclude={"tax"})
    def read_item_public_data(item_id: str):
        return items[item_id]

## 8. Annotated (should be used with at least two arguments (a type and an annotation))

### 8.1 Annotated - min_length and max_length

    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(
        q: Annotated[str | None, Query(min_length=3, max_length=50)] = None
    ):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results

### 8.2 Annotated - add regular expression

    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(
        q: Annotated[
            str | None, Query(pattern="^fixed.+")
        ] = None
    ):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results

### 8.3 Annotated - query parameter - optional

    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(q: Annotated[str | None, Query()] = None):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results


### 8.4 Annotated - query parameter - default value

    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(q: Annotated[str, Query()] = "rick"):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results

### 8.5 Annotated - query parameter - required
    
    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(q: Annotated[str, Query()]):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results

### 8.6  Annotated - query parameter - list style (add item)
    
    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(q: Annotated[list[str], Query()]):
        query_items = {"q": q}
        return query_items

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/292e0ca5-321b-440f-aaf8-22c655cf8349)

    from typing import Annotated
    from fastapi import FastAPI, Query
    
    app = FastAPI()
    
    @app.get("/items/")
    def read_items(q: Annotated[list[str], Query()] = ["foo", "bar"]):
        query_items = {"q": q}
        return query_items
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/267e8536-46c3-48ba-9683-313ca91f5ce4)

### 8.7 Annotated - path parameter

    from typing import Annotated
    from fastapi import FastAPI, Path
    
    app = FastAPI()
    
    @app.get("/items/{item_id}")
    def read_items(item_id: Annotated[int, Path()]):
        results = {"item_id": item_id}
        return results

### 8.8 Annotated - ge, gt, le, lt

    from fastapi import FastAPI, Path, Query
    app = FastAPI()
    
    @app.get("/items/{item_id}")
    def read_items(
        item_id: Annotated[int, Path(ge=0, le=1000)],
        q: Annotated[float, Query(gt=0, lt=10.5)]
    ):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4497ad96-3a9d-412d-b618-52e49ad89259)

### 8.9 Annotated - body parameter - request body

    from typing import Annotated
    from fastapi import Body, FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
    
    class User(BaseModel):
        username: str
        full_name: str | None = None
    
    @app.put("/items/{item_id}")
    def update_item(
        item_id: int,
        item: Item,
        user: User,
        importance: Annotated[int, Body(gt=0)],
        q: str | None = None
    ):
        results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
        if q:
            results.update({"q": q})
        return results

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/71e4c10b-b43a-464f-a6f8-88e7d5c4e886)


### 8.10 Annotated - body parameter - request body - Embed style

    from typing import Annotated
    from fastapi import Body, FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None
    
    @app.put("/items/{item_id}")
    def update_item(item_id: int, item: Annotated[Item, Body(embed=True)]):
        results = {"item_id": item_id, "item": item}
        return results

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7a6e25a1-6dbb-4955-a74e-22129a5b510a)


## 9. File Parameter

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/000e8397-6481-492b-bd1a-d6e4efcd10d1)

### 9.1 Request Files - File Parameter

#### 9.1.1 File Parameter with Annotated

    from typing import Annotated
    from fastapi import FastAPI, File
    
    app = FastAPI()
    
    @app.post("/files/")
    def create_file(file: Annotated[bytes, File(description="A file read as bytes")]):
        return {"file_size": len(file)}
    
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/569b6263-1884-48ab-9e5d-073d04e3483a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b6ec759d-49a3-47a3-9e08-3db3dc47ef5c)

#### 9.1.2 File Parameter with UploadFile

    from fastapi import FastAPI, UploadFile
    
    app = FastAPI()
    
    @app.post("/uploadfile/")
    def create_upload_file(file: UploadFile):
        return {"filename": file.filename}

    ########################################################

    from typing import Annotated
    from fastapi import FastAPI, File, UploadFile
    
    app = FastAPI()
    
    @app.post("/uploadfile/")
    def create_upload_file(file: Annotated[UploadFile, File(description="A file read as UploadFile")]):
        return {"filename": file.filename}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e4b71c6a-37bf-4a3a-a08f-d26ac4b32b7b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/fda018a1-4168-4371-94a7-4ca11b0fc39b)

#### 9.1.3 Multiple File Uploads

    from typing import Annotated
    from fastapi import FastAPI, File
    
    app = FastAPI()
    
    @app.post("/files/")
    async def create_files(files: Annotated[list[bytes], File(description="Multiple files as bytes")]):
        return {"file_sizes": [len(file) for file in files]}

    #######################################################################################################

    from typing import Annotated
    from fastapi import FastAPI, File, UploadFile
    
    app = FastAPI()
    
    @app.post("/uploadfiles/")
    async def create_upload_files(files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")]):
        return {"filenames": [file.filename for file in files]}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8e1c8996-4b0b-4970-a617-6d00501e81bf)

### 9.2 Response Files - File Parameter

#### 9.2.1 Return image file

    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    import matplotlib.pyplot as plt
    import io
    
    app = FastAPI()
    
    def generate_matplotlib_image():
       plt.figure(figsize=(5, 5))
       plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
       plt.title("Sample Plot")
       plt.xlabel("x")
       plt.ylabel("y")
    
       buffer = io.BytesIO()
       plt.savefig(buffer, format="png")
       buffer.seek(0)
       return buffer
    
    @app.get("/generate-image/")
    async def generate_image():
       buffer = generate_matplotlib_image()
    
       # Create a temporary file to hold the image
       # the image is saved in the same directory where your FastAPI application is running
       
       temp_file = "generated_image.png"
       with open(temp_file, "wb") as f:
           f.write(buffer.read())
    
       # Serve the file over FastAPI
       return FileResponse(temp_file, headers={"Content-Disposition": "attachment; filename=generated_image.png"})

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/254094fa-928b-4500-9c86-fa09c2c4cfe2)


## 10. Enumeration (pre-defined value in drop down style)

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

## 11. Metadata Configuration

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/074e72be-583a-4819-bd81-f1f1c2786fc5)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0c9ed9c1-9fd2-47a4-b643-856a72662cf0)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9ca2726e-9526-4388-a1cc-5c9816fc3154)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/09a65f59-0f23-4c95-abb7-1f5f94653713)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5379bddf-60a4-45c8-b62b-aaa59732f2ac)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ca18bc99-c8ff-482c-babb-97b8ac4bf4e2)


## 12. Path Operation Configuration (tags, summary, description, docstring)

### 12.1 tags

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/bcb4b983-faeb-4cce-8c54-970f1fd83c53)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/57ffcc41-8389-492e-bacf-29354974f08e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4be8828c-622b-4573-9a21-ba58ee39870b)

### 12.2 summary and description

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/baab55f6-ea0f-4dce-a39b-2619f28d3481)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/be891393-536e-4318-97b8-da931efbd9f0)

### 12.3 docstring

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/97bf94d0-a677-49b9-ae8d-b4dd5a81de5e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6f1f2d69-84a3-40d3-ad0b-1f838d07e1b9)

### 12.4 response description

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7101d4c8-ace3-4f55-b586-e04d49eef89f)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/65e6865d-e648-47e4-9e1f-f7f7344dd035)

### 12.5 Deprecate a path operation

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/771b7def-6e78-4aef-b4b0-5650c6618a65)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4f9ce4ca-ce08-4043-bc23-a585e8c21019)


## 13. Dependency Injection (depends on which function and which data type this function returned)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4e42a1b4-f42b-4fb5-94f2-d6d189797a62)

### 13.1 Simple Example

    from typing import Annotated
    
    from fastapi import Depends, FastAPI
    
    app = FastAPI()
    
    
    async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
        return {"q": q, "skip": skip, "limit": limit}
    
    
    @app.get("/items/")
    async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
        return commons
    
    
    @app.get("/users/")
    async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
        return commons

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3ff1d352-941e-40d6-b8cd-46ad5bc0807b)

### 13.2 Class as Dependency

    from fastapi import FastAPI, Depends
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    app = FastAPI()
    
    # Load the ColumnTransformer and model when the app starts
    column_transformer = joblib.load("your_column_transformer_path.joblib")
    predictive_model = joblib.load("your_model_path.joblib")
    
    # Dependencies for FastAPI
    async def get_column_transformer():
        return column_transformer
    
    async def get_predictive_model():
        return predictive_model
    
    @app.post("/predict/")
    async def predict(input_data: dict, 
                      transformer: ColumnTransformer = Depends(get_column_transformer), 
                      model: RandomForestClassifier = Depends(get_predictive_model)):
        # Preprocess the input using the ColumnTransformer
        processed_data = transformer.transform([input_data])
    
        # Make prediction using the model
        prediction = model.predict(processed_data)
    
        return {"prediction": prediction[0]}

## 14. Cookie Parameter

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0fe4b670-8af7-46f9-a169-dadd6c71a3bb)

## 15. Header Parameter

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6101dd98-e97f-4374-af79-368c2182261e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/10688545-ede0-42f5-81dc-efbf24af5330)

## 16. Form Parameter

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e27e3a70-c644-4b7d-b193-17fbf87fb0c8)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/39bd7eea-643b-4570-987d-795d65e1636b)

## 17. file structure (don't forget the init file)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a2ae6513-6f1e-4713-b5ed-e38596fab4f7)

## 18. Testing

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8d6d35ea-8b46-4054-80be-04e6a55a7fd0)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1cf92ffe-9107-4b9d-a934-140445c7e329)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c94067cf-22b2-4c32-b76f-ffc8703f1337)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f6fbfc29-6a9e-46bf-8747-abd2b4efa160)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2e75b7cc-3213-47ec-b11a-152853d21370)

    from fastapi.testclient import TestClient
    
    from .main import app
    
    client = TestClient(app)
    
    def test_read_item():
        response = client.get("/items/foo", headers={"X-Token": "coneofsilence"})
        assert response.status_code == 200
        assert response.json() == {
            "id": "foo",
            "title": "Foo",
            "description": "There goes my hero",
        }
    
    
    def test_read_item_bad_token():
        response = client.get("/items/foo", headers={"X-Token": "hailhydra"})
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid X-Token header"}
    
    
    def test_read_inexistent_item():
        response = client.get("/items/baz", headers={"X-Token": "coneofsilence"})
        assert response.status_code == 404
        assert response.json() == {"detail": "Item not found"}
    
    
    def test_create_item():
        response = client.post(
            "/items/",
            headers={"X-Token": "coneofsilence"},
            json={"id": "foobar", "title": "Foo Bar", "description": "The Foo Barters"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "id": "foobar",
            "title": "Foo Bar",
            "description": "The Foo Barters",
        }
    
    
    def test_create_item_bad_token():
        response = client.post(
            "/items/",
            headers={"X-Token": "hailhydra"},
            json={"id": "bazz", "title": "Bazz", "description": "Drop the bazz"},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid X-Token header"}
    
    
    def test_create_existing_item():
        response = client.post(
            "/items/",
            headers={"X-Token": "coneofsilence"},
            json={
                "id": "foo",
                "title": "The Foo ID Stealers",
                "description": "There goes my stealer",
            },
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Item already exists"}

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3d68fee8-b859-47f8-9b0f-d1b38c243f35)

## 19. Debugging

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/31f04d2d-c9b1-4c84-aadc-4c060791d5ff)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9f620a10-b80c-406d-bd80-b376dcf15c4b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/bdc7810e-4bcd-4c8f-aa65-56b0d3533a08)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/833b6b9d-aae5-4b10-95b8-6e6a8d81dd08)


# Backend vs Frontend

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/371e92d1-d225-48cf-8e5f-40a758de8a4e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/403d6776-cb62-4813-aa10-943d0c49e8fa)


