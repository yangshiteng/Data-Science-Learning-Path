![image](https://user-images.githubusercontent.com/60442877/204012536-46e03b5e-f029-4e57-9479-34ab77c81ebe.png)

# Solution 1:

    class Solution:
        def longestCommonPrefix(self, strs): 
            min_len = min([len(i) for i in strs])
            result = ""
            for i in range(min_len):
                temp = set([k[0:i+1] for k in strs])
                if len(temp) == 1:
                    result = list(temp)[0]
                else:
                    break
            return result
