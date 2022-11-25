![image](https://user-images.githubusercontent.com/60442877/204037407-8b5b03ab-7dec-48c0-846c-9f6c374351bc.png)

# Solution:

    class Solution:
        def plusOne(self, digits):
            digits = [str(i) for i in digits]
            result = list(str(int(''.join(digits))+1))
            return result
        
