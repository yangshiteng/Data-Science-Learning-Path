![image](https://user-images.githubusercontent.com/60442877/221387612-f95200f8-8b0e-4774-81cb-7b16cdbcd72c.png)

    txt1 = "My name is {fname}, I'm {age}".format(fname = "John", age = 36)
    txt2 = "My name is {0}, I'm {1}".format("John",36)
    txt3 = "My name is {}, I'm {}".format("John",36)
    # all return "My name is John, I'm 36"

# Formatting Types

## * Use a comma as a thousand separator

    txt = "The universe is {:,} years old."
    print(txt.format(13800000000)) 
    # return "The universe is 13,800,000,000 years old."
    
## * Use a underscore as a thousand separator

    txt = "The universe is {:_} years old."
    print(txt.format(13800000000))
    # return "The universe is 13_800_000_000 years old."
    
## * Scientific format, with a lower case e

    txt = "We have {:e} chickens."
    print(txt.format(5))
    # return "We have 5.000000e+00 chickens." 
    
## * Scientific format, with an upper case E

    txt = "We have {:E} chickens."
    print(txt.format(5))
    # return "We have 5.000000E+00."
    
## * Fix point number format

    # Use "f" to convert a number into a fixed point number, 
    # default with 6 decimals, but use a period followed by a number to specify the number of decimals:

    txt = "The price is {:.2f} dollars."
    print(txt.format(45))
    # return "The price is 45.00 dollars."

    # without the ".2" inside the placeholder, this number will be displayed like this:

    txt = "The price is {:f} dollars."
    print(txt.format(45))
    # return "The price is 45.000000 dollars."





