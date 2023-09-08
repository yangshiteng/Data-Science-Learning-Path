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

## 7. Body Parameter

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

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/157728d6-194c-4df7-b5ee-9a7727a5f91e)

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

## 11. Title, Version, Description, Tags

    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from legal_metes_and_bound.script.metesandbound_verification import verify_metesandbound
    
    app = FastAPI(title = "MetesandBound Legal Boundary Verification Service", version="1.0", description="This is a service used to verify if the boundary of a metesandbounds legal is enclosed or not")
    
    @app.post("/verify-metesandbound-enclose/",tags = ['Is Enclose'])
    async def metesandbound_enclose_verify(legal_str: str):
       
        inst_model = verify_metesandbound()
    
        is_enclose = inst_model.is_enclose(legal_str)
    
        if is_enclose == 'Enclosed':
            buffer = inst_model.buffer_return(legal_str)
    
            # Create a temporary file to hold the image
            temp_file = "generated_image.png"
            with open(temp_file, "wb") as f:
                f.write(buffer.read())
    
            # Serve the file over FastAPI
            return FileResponse(temp_file, headers={"Content-Disposition": "attachment; filename=generated_image.png"})
    
        else:
            return is_enclose

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6f08261d-f5e5-4f0f-9a36-657469a743dd)

    
# Backend vs Frontend

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/371e92d1-d225-48cf-8e5f-40a758de8a4e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/403d6776-cb62-4813-aa10-943d0c49e8fa)


