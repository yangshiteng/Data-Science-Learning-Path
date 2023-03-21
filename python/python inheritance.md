
![image](https://user-images.githubusercontent.com/60442877/226481003-9a00ad9d-0859-46f1-8bda-e6debe693476.png)

# 1. Create a Parent Class

![image](https://user-images.githubusercontent.com/60442877/226491753-76c01790-0e06-4d73-95ea-5a7e12d32057.png)

# 2. Create a Child Class

![image](https://user-images.githubusercontent.com/60442877/226493051-c33578b0-7603-4cde-9e8b-56ae9d5246ff.png)

    class Person:
      def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

      def printname(self):
        print(self.firstname, self.lastname)

    class Student(Person):
      pass

    x = Student("Mike", "Olsen")
    
    x.printname()
    # return Mike Olsen

# 3. Add the \_\_init\_\_() Function (the init function in Child class will override the init function in Parent Class)

![image](https://user-images.githubusercontent.com/60442877/226493626-7ea2f491-1d3b-4f59-b8a0-7ab9125ab60a.png)

![image](https://user-images.githubusercontent.com/60442877/226493769-42def15c-ff35-40c1-adcc-e6031f10843a.png)

    class Person:
      def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

      def printname(self):
        print(self.firstname, self.lastname)

    class Student(Person):
      def __init__(self, fname, lname):
        Person.__init__(self, fname, lname)

    x = Student("Mike", "Olsen")
    
    x.printname()
    # return Mike Olsen
    
 # 4. Use the super() function (make the child class inherit everything from its parent without specifying the parent class name)
 
 ![image](https://user-images.githubusercontent.com/60442877/226494635-c0f512f2-8c21-4be2-ba67-cc8364876778.png)

    class Person:
      def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

      def printname(self):
        print(self.firstname, self.lastname)

    class Student(Person):
      def __init__(self, fname, lname):
        super().__init__(fname, lname)

    x = Student("Mike", "Olsen")
    
    x.printname()
    # return Mike Olsen
    
# 5. Add Properties

![image](https://user-images.githubusercontent.com/60442877/226495854-13c747c5-fd69-4d7f-820c-12c14b88821e.png)

    class Person:
      def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

      def printname(self):
        print(self.firstname, self.lastname)

    class Student(Person):
      def __init__(self, fname, lname, year):
        super().__init__(fname, lname)
        self.graduationyear = year

    x = Student("Mike", "Olsen", 2019)
    
    print(x.graduationyear)
    # return 2019
    
# 6. Add Methods (the inheritance of the parent method will be overridden if the child class and parent class have the same method name)
 
![image](https://user-images.githubusercontent.com/60442877/226496186-47f669ab-56f9-46b6-9dd5-fae10aed33f0.png)

     class Person:
      def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

      def printname(self):
        print(self.firstname, self.lastname)

      def welcome(self):
        print("Welcome", self.firstname, self.lastname, "to the class")

    class Student(Person):
      def __init__(self, fname, lname, year):
        super().__init__(fname, lname)
        self.graduationyear = year

      # override the method in Parent class
      def welcome(self):
        print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

    x = Student("Mike", "Olsen", 2019)
    x.welcome()
    # return Welcome Mike Olsen to the class of 2019

 
 
 
 
 
 
 
 
