![image](https://user-images.githubusercontent.com/60442877/226229412-ec172af9-5925-4700-a5f4-bd75b6bbd350.png)

# 1. Create a Class

![image](https://user-images.githubusercontent.com/60442877/226229998-8939c4e5-4ddf-40ec-9712-7ab76ac9b32d.png)

# 2. Define an Object or an Instance with the Class Created above with specified parameter values

## the defined object or instance will have all the properties of the class (except the methods that can only be called inside class)

![image](https://user-images.githubusercontent.com/60442877/226230426-42f68d7c-1756-4416-b0b0-df83a2e14ca8.png)

# 3. The \_\_init\_\_() Function

![image](https://user-images.githubusercontent.com/60442877/226230792-337f1925-3c1a-4e0b-8c07-dba987135be5.png)

![image](https://user-images.githubusercontent.com/60442877/226230812-383a3c40-bd90-4188-9613-ff9487a24d9e.png)

## the Class parameters are defined in \_\_init\_\_() function

    class Person:
      def __init__(self, name, age):
        self.person_name = name
        self.person_age = age

    p1 = Person("John", 36)

    print(p1.person_name)
    # return John
    
    print(p1.person_age)
    # return 36

## the Class can have 0 parameter

    class Person:
      def __init__(self):
        self.person_name = 'John'
        self.person_age = 36

    p1 = Person()

    print(p1.person_name)
    # return John
    
    print(p1.person_age)
    # return 36

![image](https://user-images.githubusercontent.com/60442877/226231676-67e7ab00-49f1-4746-9e15-9c6e196b2981.png)

# 4. The \_\_str\_\_() Function

![image](https://user-images.githubusercontent.com/60442877/226232716-66303ec5-8710-4ea8-8b33-f077cec88238.png)

![image](https://user-images.githubusercontent.com/60442877/226232823-48df9843-2a41-4c2a-b898-87bdead76a63.png)

![image](https://user-images.githubusercontent.com/60442877/226232835-5cc5b07f-dbd0-4820-8207-68a2ec083f25.png)

![image](https://user-images.githubusercontent.com/60442877/226232892-b3727475-de7e-43e8-bb28-2190f95c4825.png)

![image](https://user-images.githubusercontent.com/60442877/226232903-d97bdfd2-c026-43cd-b5b8-31b1fb3c5337.png)

# 5. Object Methods

![image](https://user-images.githubusercontent.com/60442877/226233428-af2a5490-4bf7-4c65-87b5-8a2b273b6fa0.png)

![image](https://user-images.githubusercontent.com/60442877/226233450-e861cfd7-0aa4-465d-8679-1d186b396a07.png)

# 6. The self parameter

![image](https://user-images.githubusercontent.com/60442877/226234153-62810443-8edf-44af-9da3-f1cb9693072a.png)

![image](https://user-images.githubusercontent.com/60442877/226234168-f37f80c0-5105-4c8d-9de7-e45b2ba27bc5.png)

![image](https://user-images.githubusercontent.com/60442877/226234326-e702702d-4efb-45bd-9d42-63511fb579b7.png)

# 7. Modify Object's properties or attributes

![image](https://user-images.githubusercontent.com/60442877/226235106-4ccbe462-775f-4692-a021-701abe9df787.png)

![image](https://user-images.githubusercontent.com/60442877/226235132-55d56295-b692-43d3-90f3-2f4881080dde.png)

# 8. Delete Object's properties or attributes

![image](https://user-images.githubusercontent.com/60442877/226238172-2b3f386f-c417-4801-8171-45bc7c3912de.png)

![image](https://user-images.githubusercontent.com/60442877/226238190-51ceaaf8-c8d8-43dd-a9ff-842e3dfa53ea.png)

## 9. Delete Objects

![image](https://user-images.githubusercontent.com/60442877/226238306-9c1e3f7c-bde6-4a83-a5e4-e9e7d81e33da.png)

## 10. The pass statement

![image](https://user-images.githubusercontent.com/60442877/226238359-a55e6d25-7350-406a-a79a-6e802a68f1ed.png)








