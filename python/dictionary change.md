# 1. Change Items

## You can change the value of a specific item by referring to its key name

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    thisdict["year"] = 2018

    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 2018}
    
## update() - update the dictionary with the items from the given argument, the argument must be a dictionary, or an iterable object with key:value pairs

    thisdict = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    thisdict.update({"year": 2020})

    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 2020}
    
    thisdict.update({"year": 2021,"model": "Xddd"})

    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Xddd', 'year': 2021}
    
# 2. Add items

## Adding an item to the dictionary is done by using a new index key and assigning a value to it

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    thisdict["color"] = "red"
    
    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 1964, 'color': 'red'}
    
## update() - The update() method will update the dictionary with the items from a given argument. If the item does not exist, the item will be added. The argument must be a dictionary, or an iterable object with key:value pairs.

    thisdict = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    thisdict.update({"color": "red", "year": 1965})

    print(thisdict)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 1965, 'color': 'red'}





    
