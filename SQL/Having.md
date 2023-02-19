![image](https://user-images.githubusercontent.com/60442877/206933918-6951375d-c7eb-422c-bcb2-de7018ec4bbe.png)

![image](https://user-images.githubusercontent.com/60442877/206933958-16e8aaf9-7b56-4734-81ca-0d51b0a4ad1c.png)

# LeetCode Question (sum of boolean)

![image](https://user-images.githubusercontent.com/60442877/219975886-a2e95cc8-ffb0-4798-a855-3fbc5404e2da.png)
![image](https://user-images.githubusercontent.com/60442877/219975892-7f49059b-9e98-4299-a79d-5ecdde7aa415.png)

    select S.buyer_id
    from Sales S left join Product P on S.product_id = P.product_id
    group by buyer_id
    having sum(P.product_name = 'S8') > 0 and sum(P.product_name = 'iPhone') = 0
#####################################################################

    select S.buyer_id
    from Sales S left join Product P on S.product_id = P.product_id
    group by buyer_id
    having sum(if(P.product_name = 'S8',1,0)) > 0 and sum(if(P.product_name = 'iPhone',1,0)) = 0

