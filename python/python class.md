![image](https://user-images.githubusercontent.com/60442877/226229412-ec172af9-5925-4700-a5f4-bd75b6bbd350.png)

# 1. Create a Class

![image](https://user-images.githubusercontent.com/60442877/226229998-8939c4e5-4ddf-40ec-9712-7ab76ac9b32d.png)

# 2. Define a Object with the Class Created

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



