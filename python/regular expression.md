# Introduction

![image](https://user-images.githubusercontent.com/60442877/227833271-35f0be93-ed46-4c16-85d9-9d6a98667d35.png)

![image](https://user-images.githubusercontent.com/60442877/227833070-d113a7c4-7adf-403f-bf9f-a88ff3e81d6a.png)

![image](https://user-images.githubusercontent.com/60442877/227834679-ec6472be-060c-4759-80c0-d84e4ae32c12.png)

![image](https://user-images.githubusercontent.com/60442877/227835461-a093683e-2621-4f82-8725-0189d50f0ca5.png)

![image](https://user-images.githubusercontent.com/60442877/227835742-1253f0ee-7ba7-4c48-82d8-870ec73e6f1d.png)

# Methods in re

## 1. re.search() - return match object

### * If there is more than one match, only the first occurrence of the match will be returned

![image](https://user-images.githubusercontent.com/60442877/227834272-c019919b-ab37-4db8-bb33-a3a8a14a3330.png)

![image](https://user-images.githubusercontent.com/60442877/227834297-f118b93f-7c7b-4d8e-8b45-a6409eab615f.png)

## 2. re.match() - return match object

### * only checks if the pattern matches at the beginning of the string

![image](https://user-images.githubusercontent.com/60442877/227839159-5b089646-fc57-4156-b7e1-3fdf4515cd03.png)

![image](https://user-images.githubusercontent.com/60442877/227839181-19a1ab97-d136-4660-95a8-5ca5c2986a00.png)

![image](https://user-images.githubusercontent.com/60442877/227839229-bd93892e-da11-4afb-a4e4-c883904d6b51.png)

## 3. re.finditer() - return match object



## 4. re.findall()

![image](https://user-images.githubusercontent.com/60442877/227837071-c7410b33-0fdc-46f0-9ead-8326fb400d31.png)

## 5. re.split()

![image](https://user-images.githubusercontent.com/60442877/227837451-534657e3-83eb-4b60-badc-a3b195245573.png)

![image](https://user-images.githubusercontent.com/60442877/227837468-f0bbe21c-fc7c-4350-a8d7-c79a46bb7b62.png)

## 6. re.sub()

![image](https://user-images.githubusercontent.com/60442877/227838385-20fff413-9a50-474d-93ce-cc6cbb9bb9cd.png)

![image](https://user-images.githubusercontent.com/60442877/227838409-d2914a9f-4ec3-480e-bdbd-634b7e040966.png)

![image](https://user-images.githubusercontent.com/60442877/227838431-b71b2ca7-23b7-487c-aa4f-cc7abbf2d8f5.png)

![image](https://user-images.githubusercontent.com/60442877/227838450-69319b82-f244-45b4-adf7-72cd9dc9e610.png)

## 7. re.compile()

# Match Object and its methods

![image](https://user-images.githubusercontent.com/60442877/228101400-71eb7d16-83b9-4e17-9587-a8215522e8d7.png)

## 1. group() - Returns the matched substring

### * You can also pass an integer as an argument to get a specific captured group
### * group(0) returns the entire match
### * group(1) returns the first captured group, and so on

    # In this example, we're using a pattern that matches a number followed by one or more whitespace characters and a word
    
    pattern = r'(\d+)\s+(\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)
    
    print(match.group())    # Output: "42 apples"
    print(match.group(0))   # Output: "42 apples"
    print(match.group(1))   # Output: "42"
    print(match.group(2))   # Output: "apples"
    
### * We can also assign the name to each group, and retrieve it by using it name
    
    pattern = r'(?P<digit>\d+)\s+(?P<words>\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)
    
    print(match.group())    # Output: "42 apples"
    print(match.group(0))   # Output: "42 apples"
    print(match.group(1))   # Output: "42"
    print(match.group(2))   # Output: "apples"
    
    print(match.group("digit")) # Output: "42"
    print(match.group("words")) # Output: "apples"
    

## 2. groups() - Returns a tuple containing all captured groups
    
    pattern = r'(?P<digit>\d+)\s+(?P<words>\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)

    print(match.groups())   # Output: ('42', 'apples')








    import re

    # Sample text
    text = "John Smith (25) works at Example Corp."

    # Regular expression pattern with named groups
    pattern = r'(?P<name>[A-Za-z\s]+)\s\((?P<age>\d+)\)\sworks\sat\s(?P<company>[A-Za-z\s]+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Check if a match is found
    if match:
        print("Match found:")
        print("Full match:", match.group())
        print("Full match:", match.groups())
        print("Name:", match.group('name'))
        print("Age:", match.group('age'))
        print("Company:", match.group('company'))
        print("Match start:", match.start())
        print("Match end:", match.end())
        print("Match span:", match.span())
        print("Match span:", match.span('name'))
        print("Match span:", match.string)
    else:
        print("No match found")
