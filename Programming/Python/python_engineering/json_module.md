![image](https://user-images.githubusercontent.com/60442877/227817624-2ae594d1-3db0-4236-861e-c8cf9246c6d1.png)

# json.loads() - Parse JSON, Convert from JSON string to Python dictionary

![image](https://user-images.githubusercontent.com/60442877/227821019-e92ed03c-5b4a-495f-be3e-02065b64c121.png)

    import json

    # some JSON data
    x = '{ "name":"John", "age":30, "city":"New York"}'

    # parse x
    y = json.loads(x)

    # the result is a Python dictionary
    print(type(y))  # <class 'dict'>
    
    print(y["age"]) # 30


# json.dumps() - Convert from Python objects to JSON string

![image](https://user-images.githubusercontent.com/60442877/227821314-c496aec3-045d-4299-a992-ed49da8e3c15.png)

    import json

    # a Python object (dict):
    x = {
      "name": "John",
      "age": 30,
      "city": "New York"
    }

    # convert into JSON:
    y = json.dumps(x)

    # the result is a JSON string:
    print(y) # '{"name": "John", "age": 30, "city": "New York"}'
    
![image](https://user-images.githubusercontent.com/60442877/227821476-42835178-0fbe-4b54-8643-bd644f4b8c72.png)

    import json

    print(json.dumps({"name": "John", "age": 30})) 
    # '{"name": "John", "age": 30}'
    
    print(json.dumps(["apple", "bananas"]))
    # '["apple", "bananas"]'
    
    print(json.dumps(("apple", "bananas")))
    # '["apple", "bananas"]'
    
    print(json.dumps("hello"))
    # 'hello'
    
    print(json.dumps(42))
    # '42'
    
    print(json.dumps(True))
    # 'trye'
    
    print(json.dumps(False))
    # 'false'
    
    print(json.dumps(None))
    # 'null'

![image](https://user-images.githubusercontent.com/60442877/227821980-b7e72a90-cedb-4b42-b45e-36125a9c8a7b.png)

    import json

    x = {
      "name": "John",
      "age": 30,
      "married": True,
      "divorced": False,
      "children": ("Ann","Billy"),
      "pets": None,
      "cars": [
        {"model": "BMW 230", "mpg": 27.5},
        {"model": "Ford Edge", "mpg": 24.1}
      ]
    }

    # convert into JSON:
    y = json.dumps(x)

    # the result is a JSON string:
    print(y)
    # '{"name": "John", "age": 30, "married": true, "divorced": false, "children": ["Ann","Billy"], "pets": null, "cars": [{"model": "BMW 230", "mpg": 27.5}, {"model": "Ford Edge", "mpg": 24.1}]}'

![image](https://user-images.githubusercontent.com/60442877/227822454-1d5b0624-9971-44da-928c-649263ab838d.png)

![image](https://user-images.githubusercontent.com/60442877/227822476-857b39df-1c9e-4bb0-8361-a7bafd7cd539.png)

![image](https://user-images.githubusercontent.com/60442877/227822558-0e06d22c-6ead-4d0f-b078-337b6ed7a18a.png)

![image](https://user-images.githubusercontent.com/60442877/227822582-b4b4fef9-257c-4801-a14a-3818d5182bda.png)

![image](https://user-images.githubusercontent.com/60442877/227822628-3aa1c528-90e7-4c33-a487-a1c9f5199a0b.png)

![image](https://user-images.githubusercontent.com/60442877/227822648-a0fc451f-481f-4225-a65e-81e4857f51c2.png)


