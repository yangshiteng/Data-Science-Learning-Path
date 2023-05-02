# 1. Union

## union() or update() - returns a new set containing all items from both sets

![image](https://user-images.githubusercontent.com/60442877/226116474-8a394433-eedf-4351-ab5d-3d16d275623c.png)

![image](https://user-images.githubusercontent.com/60442877/226116478-75fb1b05-34c4-47b9-babf-c5fce42b4d26.png)

![image](https://user-images.githubusercontent.com/60442877/226116492-9d8570db-4459-43dc-8bef-24e33fcc55a0.png)

![image](https://user-images.githubusercontent.com/60442877/226116498-088c55ed-64ea-4885-992f-0ac02ffbc3d6.png)

![image](https://user-images.githubusercontent.com/60442877/226116505-8d9c4c85-7054-4edc-b944-b7dcee4b3a7f.png)

# 2. Intersection

## intersection_update() - keep only the items that are present in both sets

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    x.intersection_update(y)

    print(x) # return {'apple'}
    print(y) # return {"google", "microsoft", "apple"}

## intersection() - return a new set, that only contains the items that are present in both sets

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    z = x.intersection(y)

    print(z) # return {'apple'}
    print(x) # return {"apple", "banana", "cherry"}
    print(y) # return {"google", "microsoft", "apple"}

# 3. Symmetric difference

## symmetric_difference_update() - keep only the elements that are NOT present in both sets

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    x.symmetric_difference_update(y)

    print(x) # return {'banana', 'microsoft', 'google', 'cherry'}
    print(y) # return {"google", "microsoft", "apple"}

## symmetric_difference() - return a new set, that contains only the elements that are NOT present in both sets

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    z = x.symmetric_difference(y)

    print(z) # return {'google', 'banana', 'microsoft', 'cherry'}
    print(x) # return {"apple", "banana", "cherry"}
    print(y) # return {"google", "microsoft", "apple"}

# 4. Difference

## difference_update() - keep only the elements that are NOT present in another set

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    x.difference_update(y) 

    print(x) # return {'cherry', 'banana'}
    print(y) # return {"google", "microsoft", "apple"}

## difference() - eturn a new set, that contains only the elements that are NOT present in another set

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}

    z = x.difference(y) 

    print(z) # return {'cherry', 'banana'}
    print(x) # return {"apple", "banana", "cherry"}
    print(y) # return {"google", "microsoft", "apple"}

# 5. isdisjoint() - check if two sets are disjoint or not

    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "facebook"}

    z = x.isdisjoint(y) 

    print(z) # return True

## 6. issubset() - check if the first set is the subset of the second set

    x = {"a", "b", "c"}
    y = {"f", "e", "d", "c", "b", "a"}

    z = x.issubset(y) 

    print(z) # return True
    
## 7. issuperset() - check if the first set is the superset of the second set

    x = {"f", "e", "d", "c", "b", "a"}
    y = {"a", "b", "c"}

    z = x.issuperset(y) 

    print(z) # return True


