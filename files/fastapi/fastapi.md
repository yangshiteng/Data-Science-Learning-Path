# Tutorial

https://fastapi.tiangolo.com/

https://fastapi.tiangolo.com/tutorial/

# Commonly used Code

## 1. install the fast api

    # install it with all the optional dependencies and features
    # that also includes uvicorn, that you can use as the server that runs your code
    pip install "fastapi[all]"

    # you can also install it part by part
    pip install fastapi
    pip install "uvicorn[standard]"

## 2. launch the server
    # with reload, you don't need to restart the API if you made any change 
    uvicorn main:app --reload
