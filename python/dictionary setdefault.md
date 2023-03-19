![image](https://user-images.githubusercontent.com/60442877/226189877-67c80707-703b-4b97-bf6c-65d0643265fd.png)

# self not changed and new value generated method

    car = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    x = car.setdefault("model", "Bronco")

    print(x)
    # return Mustang
    
    print(car)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 1964}
    
# self changed and new value generated method

    car = {
      "brand": "Ford",
      "model": "Mustang",
      "year": 1964
    }

    x = car.setdefault("fruit", "banana")

    print(x)
    # return banana
    
    print(car)
    # return {'brand': 'Ford', 'model': 'Mustang', 'year': 1964, 'fruit': 'banana'}
