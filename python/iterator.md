![image](https://user-images.githubusercontent.com/60442877/226499210-6baa7d32-9cb3-4745-b40f-de0e9d821279.png)

![image](https://user-images.githubusercontent.com/60442877/226499255-3e9c1062-1ac9-4c97-9d41-88290f3c2746.png)

# 1. Create an Iterator

![image](https://user-images.githubusercontent.com/60442877/226499920-712d9786-335e-41be-a284-2c181bc356f7.png)

![image](https://user-images.githubusercontent.com/60442877/226499937-fec15f47-0a14-486d-9006-c9336d8f48b1.png)

![image](https://user-images.githubusercontent.com/60442877/226500178-52334a94-9665-411b-b354-60de0a0fc765.png)

    class MyNumbers:

      def __init__(self, number):
        self.number = number

      def __iter__(self):
        self.a = self.number
        return self

      def __next__(self):
        x = self.a
        self.a += 1
        return x

    myclass = MyNumbers(5)
    myiter = iter(myclass)

    print(next(myiter)) # 5
    print(next(myiter)) # 6
    print(next(myiter)) # 7
    print(next(myiter)) # 8
    print(next(myiter)) # 9
    
# 2. Stop Iteration

![image](https://user-images.githubusercontent.com/60442877/226500321-3c69e99a-825e-4fcf-8d1d-f85c0748a1f8.png)

![image](https://user-images.githubusercontent.com/60442877/226500348-80c71188-1a06-4fc0-a549-b43d7f192e24.png)

![image](https://user-images.githubusercontent.com/60442877/226500358-c44f6524-bf26-4156-913d-0643fb757297.png)






    
