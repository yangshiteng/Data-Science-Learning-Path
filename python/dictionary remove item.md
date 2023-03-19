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
    
# del keyword 

## The del keyword removes the item with the specified key name

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    del thisdict["model"]
    
    print(thisdict)
    # return {'brand': 'Ford', 'year': 1964}

## The del keyword can also delete the dictionary completely

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    del thisdict
    print(thisdict) #this will cause an error because "thisdict" no longer exists.
    
# clear() - empties the dictionary

## self changed and no value returned method

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    thisdict.clear()
    
    print(thisdict)
    # return {}
