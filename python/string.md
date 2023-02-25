
# Multiline Strings

![image](https://user-images.githubusercontent.com/60442877/221096022-93f00053-744f-4e4a-877f-f56f2b4043b7.png)

# String Methods

## Upper, Lower or Capitalize

    a = "Hello, World!"
    print(a.upper()) # returns "HELLO, WORLD"
    
    a = "Hello, World!"
    print(a.lower()) # returns "hello, world"
    
    txt = "hello, and welcome to my world."
    x = txt.capitalize()
    print (x) # returns "Hello, and welcome to my world."

## Remove Whitespace before or/and after the text

    a = " Hello, World! "
    print(a.strip()) # returns "Hello, World"
    print(a.rstrip() # returns " Hello, World"
    print(a.lstrip() # returns "Hello, World "
    
## String Replace

    a = "Hello, World!"
    print(a.replace("H", "J")) # returns "Jello, World"
    
## String Split (default separator is space)

    a = "Hello, World!"
    print(a.split(",")) # returns ['Hello', ' World!']
    print(a.split())    # returns ['Hello,', 'World!']
