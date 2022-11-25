![image](https://user-images.githubusercontent.com/60442877/203884411-94f261dd-7edd-49aa-b414-de09c8257c90.png)

![image](https://user-images.githubusercontent.com/60442877/203884426-e24e8ac9-cb26-4a24-8f31-8e0503094cb6.png)

# Solution:

    class Solution:
        def romanToInt(self, s):
            dict1 = {}
            dict1['I'] = 1
            dict1['V'] = 5
            dict1['X'] = 10
            dict1['L'] = 50
            dict1['C'] = 100
            dict1['D'] = 500
            dict1['M'] = 1000

            dict2 = {}
            dict2['IV'] = 4
            dict2['IX'] = 9
            dict2['XL'] = 40
            dict2['XC'] = 90
            dict2['CD'] = 400
            dict2['CM'] = 900

            result = 0

            for i,j in dict2.items():
                result = result + s.count(i)*j
                s = s.replace(i,'')

            for i in list(s):
                result = result + dict1[i]

            return result
