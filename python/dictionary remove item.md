# pop() - removes the item with the specified key name 

## self changed and new value generated method, returns the removed value 

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    x = thisdict.pop("model")
    
    print(x)
    # return Mustang
    
    print(thisdict)
    # return {'brand': 'Ford', 'year': 1964}

# popitem() - removes the last inserted item (in versions before 3.7, a random item is removed instead)

## self changed and new value generated method, returns a tuple containing removed key and value 

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    x = thisdict.popitem()
    
    print(x)
    # return ('year', 1964)
    
    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Mustang'}
    
    
    
    
