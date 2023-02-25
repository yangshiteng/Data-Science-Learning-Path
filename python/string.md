
# Multiline Strings

![image](https://user-images.githubusercontent.com/60442877/221096022-93f00053-744f-4e4a-877f-f56f2b4043b7.png)

# String Methods

## Upper, Lower or Capitalize

    a = "Hello, World!"
    print(a.upper()) # return "HELLO, WORLD"
    
    a = "Hello, World!"
    print(a.lower()) # return "hello, world"
    
    txt = "hello, and welcome to my world."
    x = txt.capitalize()
    print (x) # return "Hello, and welcome to my world."

## Remove Whitespace before or/and after the text

    a = " Hello, World! "
    print(a.strip()) # return "Hello, World"
    print(a.rstrip() # return " Hello, World"
    print(a.lstrip() # return "Hello, World "
    
## String Replace

    a = "Hello, World!"
    print(a.replace("H", "J")) # returns "Jello, World"
    
## String Split (default separator is space)

    a = "Hello, World!"
    print(a.split(",")) # return ['Hello', ' World!']
    print(a.split())    # return ['Hello,', 'World!']
    
## Count

![image](https://user-images.githubusercontent.com/60442877/221363540-2b414849-61de-4a4a-bdd7-1bafe2f67561.png)

    txt = "I love apples, apple are my favorite fruit"
    x = txt.count("apple")
    print(x) # return 2
    
    txt = "I love apples, apple are my favorite fruit"
    x = txt.count("apple", 10, 24)
    print(x) # return 1
    
## Start or End with

![image](https://user-images.githubusercontent.com/60442877/221363669-84d0c7f2-244a-4269-a5f1-b9c34317c091.png)
![image](https://user-images.githubusercontent.com/60442877/221363716-1aae3765-c18c-4127-b4bf-f4ce2f2f50ff.png)

    txt = "Hello, welcome to my world."
    x = txt.startswith("Hello")
    print(x) # return True
   
    txt = "Hello, welcome to my world."
    x = txt.startswith("wel", 7, 20)
    print(x) # return True

    txt = "Hello, welcome to my world."
    x = txt.endswith(".")
    print(x) # return True
    
    txt = "Hello, welcome to my world."
    x = txt.endswith("my world.")
    print(x) # return True
    
    txt = "Hello, welcome to my world."
    x = txt.endswith("my world.", 5, 11)
    print(x) # return False

## Find the index of the first occurence of the specified value

### find()

![image](https://user-images.githubusercontent.com/60442877/221364100-b8c3ad3d-ea23-4dca-a664-ca2d31eaa3ff.png)

    txt = "Hello, welcome to my world."
    x = txt.find("e")
    print(x) # return 1

    txt = "Hello, welcome to my world."
    x = txt.find("e", 5, 10)
    print(x) # return 8
    
    txt = "Hello, welcome to my world."
    print(txt.find("q")) # return -1
    print(txt.index("q")) # return error

### index()
    
