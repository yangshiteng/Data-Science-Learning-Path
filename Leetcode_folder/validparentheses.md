![image](https://user-images.githubusercontent.com/60442877/204031202-629ee0c0-cab0-46ce-9542-a2fd5972db40.png)

### Example 4:

#### Input: s = "{[()]}"
#### Output: True

# Solution:

    class Solution(object):
      def isValid(self, s):
                d = {'(':')', '{':'}','[':']'}
                stack = []
                for i in s:
                    if i in d:  
                        stack.append(i)
                    elif len(stack) == 0 or d[stack.pop()] != i: 
                        return False
                return len(stack) == 0 
