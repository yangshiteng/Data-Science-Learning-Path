![image](https://user-images.githubusercontent.com/60442877/206933087-afe31cc4-dfb5-42f9-9766-ef083fd5b56f.png)

![image](https://user-images.githubusercontent.com/60442877/206933193-723d8183-70f3-49fb-b4ae-baeb8fde8b7a.png)

# LeetCode Question (group vs group is not allowed)

![image](https://user-images.githubusercontent.com/60442877/220369884-a4b9d659-89f4-45ef-97fe-09dbe1403d52.png)
![image](https://user-images.githubusercontent.com/60442877/220369912-d4852ae5-879b-4ec8-98d7-3a47130b54ff.png)

    with temp1 as(
    select *, dense_rank() over(partition by player_id order by event_date) as rank_temp
    from Activity),
    temp2 as(
        select event_date as install_dt, player_id
        from temp1
        where rank_temp = 1
    ),
    temp3 as(
        select *,
        if((player_id, adddate(install_dt,1)) in (select player_id, event_date from Activity),1,0) as count_num
        from temp2
    )

    select install_dt, count(player_id) as installs, round(sum(count_num)/count(player_id),2) as Day1_retention
    from temp3
    group by install_dt

* __In group by, a single record is a group, so, group vs group is not allowed__

![image](https://user-images.githubusercontent.com/60442877/220374460-7378b066-0f2a-47c2-aa11-6c63b28dc257.png)

# LeetCode Question (Filter First, Group Later)

![image](https://user-images.githubusercontent.com/60442877/220408343-25b5765d-f4fa-4fdc-941b-385e492fbf3f.png)
![image](https://user-images.githubusercontent.com/60442877/220408378-0b249604-dd3e-41a8-8533-2f1ac98986fe.png)

    with temp1 as (select *, avg(occurences) over(partition by event_type) as ave_event
    from Events)

    select business_id
    from temp1
    where occurences > ave_event
    # Use where clause to filter table first, then group
    group by business_id
    having count(event_type) > 1

# LeetCode Question (Group Comparison in Select Clause is Value-Based)

![image](https://user-images.githubusercontent.com/60442877/220463601-7f4266ac-cfb3-44ff-89e1-edff1c4302be.png)
![image](https://user-images.githubusercontent.com/60442877/220463627-8afd09bf-b36b-46cd-b008-85c929c79306.png)

    with temp1 as(
    select spend_date, user_id, 'both' as new_platform, if(count(platform)=2, sum(amount), 0) as new_amount, if(count(platform)=2, 1, 0) as user_count
    from Spending
    group by spend_date, user_id
    union
    select spend_date, user_id, 'mobile' as new_platform, if(sum(platform='mobile') = 1 and count(platform)=1, sum(amount), 0) as new_amount,  if(sum(platform='mobile')/count(platform)=1, 1, 0) as user_count
    from Spending
    group by spend_date, user_id
    union
    select spend_date, user_id, 'desktop' as new_platform, if(sum(platform='desktop')/count(platform)=1, sum(amount), 0) as new_amount, if(sum(platform='desktop')/count(platform)=1, 1, 0) as user_count
    from Spending
    group by spend_date, user_id)

    select spend_date, new_platform as platform, sum(new_amount) as total_amount, sum(user_count) as total_users
    from temp1
    group by spend_date, new_platform

![image](https://user-images.githubusercontent.com/60442877/220464584-ed8b97e2-df1a-48c9-84e8-1289670b5d24.png)


# LeetCode Question (Group with If function)

![image](https://user-images.githubusercontent.com/60442877/220686974-971f24df-8a6f-45d7-a740-7d54b3bfb6fa.png)
![image](https://user-images.githubusercontent.com/60442877/220687055-c1aea52a-8b6c-464c-bcdd-20bcc4b364fa.png)

        select id, 
        sum(if(month = 'Jan', revenue, null)) as Jan_Revenue, 
        sum(if(month = 'Feb', revenue, null)) as Feb_Revenue,
        sum(if(month = 'Mar', revenue, null)) as Mar_Revenue,
        sum(if(month = 'Apr', revenue, null)) as Apr_Revenue,
        sum(if(month = 'May', revenue, null)) as May_Revenue,
        sum(if(month = 'Jun', revenue, null)) as Jun_Revenue,
        sum(if(month = 'Jul', revenue, null)) as Jul_Revenue,
        sum(if(month = 'Aug', revenue, null)) as Aug_Revenue,
        sum(if(month = 'Sep', revenue, null)) as Sep_Revenue,
        sum(if(month = 'Oct', revenue, null)) as Oct_Revenue,
        sum(if(month = 'Nov', revenue, null)) as Nov_Revenue,
        sum(if(month = 'Dec', revenue, null)) as Dec_Revenue
        from Department
        group by id
        order by id

* Here, if(month = 'Jan', revenue, null) is (8000,null,null)



