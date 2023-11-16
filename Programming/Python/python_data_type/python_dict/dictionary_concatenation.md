

# 1. update() 
## self changed, no new value generated

![image](https://user-images.githubusercontent.com/60442877/226208659-df1b4246-300f-4979-923e-adf7c3f35c5e.png)


# 2. dictionary unpacking operator **

![image](https://user-images.githubusercontent.com/60442877/226208957-f3a1d06f-75ca-425e-a7d7-4fc2e8e1d318.png)


# Note that if there are common keys in the two dictionaries, the values from the second dictionary will overwrite the values from the first dictionary

    a = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    b = {'d': 5, 'e': 6}
    
    new_dict = {**a,**b}
    
    print({new_dict)
    # return {'a': 1, 'b': 2, 'c': 3, 'd': 5, 'e': 6}
    
    
    
