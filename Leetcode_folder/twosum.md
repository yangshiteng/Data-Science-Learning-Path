![image](https://user-images.githubusercontent.com/60442877/203883465-6c41cb5c-9545-4699-88a1-1a0e904f227f.png)

# Solution: (set a dictionary for storage)

    class Solution:
        def twoSum(self, nums, target):
            temp_df = {}
            for i,j in enumerate(nums):
                k = target - j
                if k in temp_df:
                    return [temp_df[k], i]
                temp_df[j] = i
