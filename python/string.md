
# Multiline Strings

![image](https://user-images.githubusercontent.com/60442877/221096022-93f00053-744f-4e4a-877f-f56f2b4043b7.png)

# String Methods

## 1. String Format Change

### * Upper, Lower or Capitalize

    a = "Hello, World!"
    print(a.upper()) # return "HELLO, WORLD"
    
    a = "Hello, World!"
    print(a.lower()) # return "hello, world"
    
    txt = "hello, and welcome to my world."
    x = txt.capitalize()
    print (x) # return "Hello, and welcome to my world."


### * Remove Whitespace before or/and after the text

    a = " Hello, World! "
    print(a.strip()) # return "Hello, World"
    print(a.rstrip() # return " Hello, World"
    print(a.lstrip() # return "Hello, World "


## 2. String Modification

### * String Replace

    a = "Hello, World!"
    print(a.replace("H", "J")) # return "Jello, World"
    
### * String Split (splits a string into a list) (default separator is space)

![image](https://user-images.githubusercontent.com/60442877/221386460-45cca7d1-1e3c-4bce-9926-094a8e564f55.png)

    a = "Hello World!"
    print(a.split()) # return ['Hello', 'World!']
    
    txt = "hello, my name is Peter, I am 26 years old"
    x = txt.split(", ")
    print(x) # return ['hello', 'my name is Peter', 'I am 26 years old']
    
    txt = "apple#banana#cherry#orange"
    x = txt.split("#")
    print(x) # ['apple', 'banana', 'cherry', 'orange']

### * join() - Converts the elements of an iterable into a string

    myTuple = ("John", "Peter", "Vicky")
    x = "#".join(myTuple)
    print(x) # return "John#Peter#Vicky"

    myTuple = ("John", "Peter", "Vicky")
    x = ",".join(myTuple)
    print(x) # return "John,Peter,Vicky"

## 3. String Count and Search

### * Count a specific value in the string

![image](https://user-images.githubusercontent.com/60442877/221363540-2b414849-61de-4a4a-bdd7-1bafe2f67561.png)

    txt = "I love apples, apple are my favorite fruit"
    x = txt.count("apple")
    print(x) # return 2
    
    txt = "I love apples, apple are my favorite fruit"
    x = txt.count("apple", 10, 24)
    print(x) # return 1
    
### * Find the index of the first occurence of the specified value

#### * find()

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

#### * index()

![image](https://user-images.githubusercontent.com/60442877/221386395-e9d5dab1-6d21-41a6-adbd-7f47c7405496.png)


    txt = "Hello, welcome to my world."
    x = txt.index("e")
    print(x) # return 1

    txt = "Hello, welcome to my world."
    print(txt.find("q")) # return -1
    print(txt.index("q")) # return error
    
## 4. String Checking
    
### * Check if string start or end with a specific value

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

### * Check if all characters in the string are alphanumeric (0-9A-Za-z)

![image](https://user-images.githubusercontent.com/60442877/221390177-f372e866-aace-4e90-8369-5ce72ae43411.png)

    txt = "Company 12"
    
    x = txt.isalnum()

    print(x) # return False
    
### * Check if all characters in the string are in the alphabet (a-zA-Z)

    txt = "CompanyX"

    x = txt.isalpha()

    print(x) # return True

### * Check if all characters in the string are digits

    txt = "565543"

    x = txt.isnumeric()

    print(x) # return True

### * Check if all characters in the string are whitespaces

    txt = "   "

    x = txt.isspace()

    print(x) # return True

### * Check if the string follows the rules of a title

    txt = "Hello, And Welcome To My World!"
    x = txt.istitle()
    print(x) # return True

    txt = "Hello, and Welcome to My World!"
    x = txt.istitle()
    print(x) # return False
    
### * Check if all the characters are in upper case
    
    txt = "THIS IS NOW!"

    x = txt.isupper()

    print(x) # return True
    
### * Check if all characters in the string are lower case

    txt = "hello world!"

    x = txt.islower()

    print(x) # return True

    
    
    
