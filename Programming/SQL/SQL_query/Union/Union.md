![image](https://user-images.githubusercontent.com/60442877/206932861-e8506e7c-1b5b-40fb-ac97-f2b76464c6dc.png)

![image](https://user-images.githubusercontent.com/60442877/206932907-be9521ab-07f7-432b-a168-9d85849d1e0e.png)

# Leetcode Question

![image](https://user-images.githubusercontent.com/60442877/217399338-92cebf66-24ab-4a51-a561-42706eb7f81b.png)
![image](https://user-images.githubusercontent.com/60442877/217399368-5212020b-5b5a-4f0d-8d24-00d2412598ac.png)


    with temp1 as(
    select requester_id as id, count(accepter_id) as num1
    from RequestAccepted
    group by requester_id
    union all
    select accepter_id as id, count(requester_id) as num1
    from RequestAccepted
    group by accepter_id),
    temp2 as(
    select id, sum(num1) as num from temp1 group by id
    )

    select id, num
    from temp2
    having num = (select max(num) from temp2)
