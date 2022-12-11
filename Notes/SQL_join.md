![image](https://user-images.githubusercontent.com/60442877/206078697-0a9ce00e-d215-49b5-abf1-c08dc23ad625.png)

# Inner Join

![image](https://user-images.githubusercontent.com/60442877/206081234-8b75df75-9df9-4568-a5d9-8171f4733e29.png)

![image](https://user-images.githubusercontent.com/60442877/206081390-77909326-ea00-4aba-aea0-a484b10dfcb0.png)

# Left Join

![image](https://user-images.githubusercontent.com/60442877/206081449-6505b1f2-873f-4e79-a02e-17ff25f46f01.png)
![image](https://user-images.githubusercontent.com/60442877/206081538-979aab8f-9aac-4f26-81f2-9894b1d26315.png)

# Right Join

![image](https://user-images.githubusercontent.com/60442877/206081613-f7532db5-34b9-40d4-ba74-df82a1b45aa4.png)

# Full Join

![image](https://user-images.githubusercontent.com/60442877/206081727-c868797b-3f9a-44a7-9d6f-9b88c65fa24a.png)
![image](https://user-images.githubusercontent.com/60442877/206081883-c1025445-4882-40ff-b771-7a0cad5ff244.png)

# Self Join (Cross Join the same table)

![image](https://user-images.githubusercontent.com/60442877/206083391-6fee0035-e91f-4249-bbe1-83fcde0db2c9.png)

![image](https://user-images.githubusercontent.com/60442877/206084227-efedc91e-ac38-4028-8db6-1b8183137a95.png)

![image](https://user-images.githubusercontent.com/60442877/206084455-111678ed-a0b2-448b-bcf2-bc8145ab3252.png)

# Cross Join (or called Cartesian Join)

![image](https://user-images.githubusercontent.com/60442877/206084737-213099f5-7c51-494d-99c1-612f18ab39e1.png)

![image](https://user-images.githubusercontent.com/60442877/206084227-efedc91e-ac38-4028-8db6-1b8183137a95.png)

![image](https://user-images.githubusercontent.com/60442877/206084258-5e505910-2b7e-4639-a596-543411c355e6.png)

![image](https://user-images.githubusercontent.com/60442877/206084773-960f7623-01be-41f5-bc37-a54963d85511.png)

![image](https://user-images.githubusercontent.com/60442877/206084797-27a2504a-75e5-4975-b512-de7a9dfce272.png)


# LeetCode Questions and Answers

## Left Join and simplified join keyword

![image](https://user-images.githubusercontent.com/60442877/205422907-9fe5bde5-a90c-496a-9a7e-aaf7faca4264.png)

![image](https://user-images.githubusercontent.com/60442877/205422912-4726f3fb-06fe-40bf-8856-be922b401e03.png)

    select firstName, lastName, city, state
    from Person left join Address using (personId)
