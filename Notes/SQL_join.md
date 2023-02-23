![image](https://user-images.githubusercontent.com/60442877/206078697-0a9ce00e-d215-49b5-abf1-c08dc23ad625.png)

# Inner Join

![image](https://user-images.githubusercontent.com/60442877/206081234-8b75df75-9df9-4568-a5d9-8171f4733e29.png)

![image](https://user-images.githubusercontent.com/60442877/206081390-77909326-ea00-4aba-aea0-a484b10dfcb0.png)

# Left Join

![image](https://user-images.githubusercontent.com/60442877/206081449-6505b1f2-873f-4e79-a02e-17ff25f46f01.png)
![image](https://user-images.githubusercontent.com/60442877/206081538-979aab8f-9aac-4f26-81f2-9894b1d26315.png)

# Right Join

![image](https://user-images.githubusercontent.com/60442877/206081613-f7532db5-34b9-40d4-ba74-df82a1b45aa4.png)

# Full Join (not applicable in MySQL)

![image](https://user-images.githubusercontent.com/60442877/206081727-c868797b-3f9a-44a7-9d6f-9b88c65fa24a.png)
![image](https://user-images.githubusercontent.com/60442877/206081883-c1025445-4882-40ff-b771-7a0cad5ff244.png)

# LeetCode Question

## Left Join and simplified join keyword

![image](https://user-images.githubusercontent.com/60442877/205422907-9fe5bde5-a90c-496a-9a7e-aaf7faca4264.png)

![image](https://user-images.githubusercontent.com/60442877/205422912-4726f3fb-06fe-40bf-8856-be922b401e03.png)

    select firstName, lastName, city, state
    from Person left join Address using (personId)

## Filter First and Join Later vs Join First Filter Later

![image](https://user-images.githubusercontent.com/60442877/220133724-79242e4a-dab7-45d7-9678-62c0f50d3c87.png)
![image](https://user-images.githubusercontent.com/60442877/220133746-5e99e520-7126-473f-8ac7-b2e37ce22a17.png)

    select 
    b.book_id,
    b.name
    from 
    (select * from books where available_from <= "2019-05-23") b 
    left join (select * from orders where dispatch_date >= "2018-06-23") o
    on b.book_id=o.book_id 
    group by b.book_id,b.name
    having sum(o.quantity) is null or sum(quantity)<10
![image](https://user-images.githubusercontent.com/60442877/220134626-47bbbd29-70d3-4abc-b773-77dde34434d0.png)

![image](https://user-images.githubusercontent.com/60442877/220811771-fbcad1e1-2c9f-47c5-89e6-6b8460e3e4c4.png)
![image](https://user-images.githubusercontent.com/60442877/220811791-54b9a61d-ee14-47a5-8f93-431ff656d61c.png)
![image](https://user-images.githubusercontent.com/60442877/220811815-f6ffaa34-6238-4a24-9712-ddc276ee7403.png)
