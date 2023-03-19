# 1. Access Items

## You can access the items of a dictionary by referring to its key name, inside square brackets

    thisdict = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    x = thisdict["model"] 
    print(x)
    # return Mustang

## get() - give you the same resul

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    x = thisdict.get("model")
    print(x)
    # return Mustang
    
# 2. Get Keys

## keys() - return a list of all the keys in the dictionary

    thisdict = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    x = thisdict.keys()

    print(x)
    # return dict_keys(['brand', 'model', 'year'])
    
## The list of the keys is a view of the dictionary, meaning that any changes done to the dictionary will be reflected in the keys list

    car = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
    }

    x = car.keys()

    print(x) #before the change
    # return dict_keys(['brand', 'model', 'year'])

    car["color"] = "white"

    print(x) #after the change
    # return dict_keys(['brand', 'model', 'year', 'color'])
    
# 3. Get Values

## values() - return a list of all the values in the dictionary


    thisdict = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    x = thisdict.values()

    print(x)
    # return dict_values(['Ford', 'Mustang', 1964])

## The list of the values is a view of the dictionary, meaning that any changes done to the dictionary will be reflected in the values list

    car = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
    }

    x = car.values()

    print(x) #before the change
    # return dict_values(['Ford', 'Mustang', 1964])

    car["year"] = 2020

    print(x) #after the change
    # return dict_values(['Ford', 'Mustang', 2020])

# 4. Get Items

## items() - return each item in a dictionary, as tuples in a list

        thisdict = {
          "brand": "Ford",
          "model": "Mustang",
          "year": 1964
        }

        x = thisdict.items()

        print(x)
        # return dict_items([('brand', 'Ford'), ('model', 'Mustang'), ('year', 1964)])

## The returned list is a view of the items of the dictionary, meaning that any changes done to the dictionary will be reflected in the items list

        car = {
        "brand": "Ford",
        "model": "Mustang",
        "year": 1964
        }

        x = car.items()

        print(x) #before the change
        # return dict_items([('brand', 'Ford'), ('model', 'Mustang'), ('year', 1964)])

        car["year"] = 2020

        print(x) #after the change
        # return dict_items([('brand', 'Ford'), ('model', 'Mustang'), ('year', 2020)])

