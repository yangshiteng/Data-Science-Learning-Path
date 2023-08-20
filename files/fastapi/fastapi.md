# Tutorial

https://fastapi.tiangolo.com/

https://fastapi.tiangolo.com/tutorial/

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a4c53be2-ec4e-437d-aeb7-1b31ee04f00d)

# Commonly used Code

## 1. Install the fast api

    # install it with all the optional dependencies and features
    # that also includes uvicorn, that you can use as the server that runs your code
    pip install "fastapi[all]"

    # you can also install it part by part
    pip install fastapi
    pip install "uvicorn[standard]"

## 2. Run the live server

    # with reload, you don't need to restart the API if you made any change 
    uvicorn main:app --reload

    # the URL where your app is being served, in your local machine
    http://127.0.0.1:8000

    # automatic interactive API documentation (provided by Swagger UI)
    http://127.0.0.1:8000/docs

    # alternative automatic documentation (provided by ReDoc)
    http://127.0.0.1:8000/redoc















    
