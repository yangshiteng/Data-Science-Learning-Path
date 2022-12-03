# Left Join








# LeetCode Questions and Answers

## 1. Left Join and simplified join keyword

![image](https://user-images.githubusercontent.com/60442877/205422907-9fe5bde5-a90c-496a-9a7e-aaf7faca4264.png)

![image](https://user-images.githubusercontent.com/60442877/205422912-4726f3fb-06fe-40bf-8856-be922b401e03.png)

    select firstName, lastName, city, state
    from Person left join Address using (personId)
