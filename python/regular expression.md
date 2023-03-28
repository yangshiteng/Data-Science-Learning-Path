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

![image](https://user-images.githubusercontent.com/60442877/228106291-51ff6327-18cc-402c-b358-493b8532cb57.png)

![image](https://user-images.githubusercontent.com/60442877/228106315-37c3a068-e95a-44f4-8b3c-a662872bed45.png)


# Match Object and its methods

![image](https://user-images.githubusercontent.com/60442877/228101400-71eb7d16-83b9-4e17-9587-a8215522e8d7.png)

## 1. group() - Returns the matched substring

### * You can also pass an integer as an argument to get a specific captured group

    # In this example, we're using a pattern that matches a number followed by one or more whitespace characters and a word
    
    pattern = r'(\d+)\s+(\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)
    
    # returns the entire match
    print(match.group())    # Output: "42 apples"
    print(match.group(0))   # Output: "42 apples"
    
    # returns the nth captured group (1-indexed)
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


## 3. start(), end(), span()

![image](https://user-images.githubusercontent.com/60442877/228103439-735645c0-2eff-4297-8ba6-64de93847313.png)

    pattern = r'(?P<digit>\d+)\s+(?P<words>\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)

    print(match.start())        # Output: 7
    print(match.start("digit")) # Output: 7
    print(match.start("words")) # Output: 10
    
    print(match.end())          # Output: 16
    print(match.end("digit"))   # Output: 9
    print(match.end("words"))   # Output: 16
    
    print(match.span())         # Output: (7, 16)
    print(match.span("digit"))  # Output: (7, 9)
    print(match.span("words"))  # Output: (10, 16)

## 4. string - return the input string

    pattern = r'(?P<digit>\d+)\s+(?P<words>\w+)'
    
    text = "I have 42 apples and 3 oranges."
    
    match = re.search(pattern, text)
    
    print(match.string)     # Output: "I have 42 apples and 3 oranges."
