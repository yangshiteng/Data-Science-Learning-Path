# self changed, no new value generated

    fruits = ['apple', 'banana', 'cherry']

    fruits.remove("banana")
    
    print(fruits) # return ['apple', 'cherry']


# self changed, new value generated

    fruits = ['apple', 'banana', 'cherry']

    x = fruits.pop(1)

    print(x) # return banana
    print(fruits) # return ['apple', 'cherry']
