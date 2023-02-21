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




