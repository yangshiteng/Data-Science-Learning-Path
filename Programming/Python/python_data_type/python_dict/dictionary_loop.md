
# loop keys 

## method 1

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    for x in thisdict:
      print(x)
      
     # brand
     # model
     # year
     
 ## method 2
 
     thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    for x in thisdict.keys():
      print(x)

     # brand
     # model
     # year

# loop values

## method 1

    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    for x in thisdict:
      print(thisdict[x])
      
     # Ford
     # Mustang
     # 1964
     
 ## method 2
 
    thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    for x in thisdict.values():
      print(x)
 
     # Ford
     # Mustang
     # 1964
     
 # loop both keys and values
 
     thisdict =	{
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }
    
    for x, y in thisdict.items():
      print(x, y)

    # brand Ford
    # model Mustang
    # year 1964
 
      
 
