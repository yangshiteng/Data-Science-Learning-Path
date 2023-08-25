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

## 2. Run the live server

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0cbf93db-e582-4c83-aad6-2319ddcb3706)

    # file name must be main.py
    # also make sure you are in the same folder with your FastAPI app instance
    # with reload, you don't need to restart the API if you made any change 
    
    uvicorn main:app --reload


















    
