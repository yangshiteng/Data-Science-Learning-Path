# Window Function

* Ranking the rows
* Calculating a single value such as sum, average, min, max, count, and assign this value to each row
* Calculating cumulative values such as sum, average
* Calculating cumulative values such as sum, average within some range


![image](https://user-images.githubusercontent.com/60442877/214994878-6a118c70-e503-4da2-840f-0fe802313aff.png)
![image](https://user-images.githubusercontent.com/60442877/214994989-7a38b4d6-d808-4624-85ff-e97ed0a6cc80.png)
![image](https://user-images.githubusercontent.com/60442877/214995016-e16c5a51-ee14-46ad-a0bf-4c2cddeedc54.png)
![image](https://user-images.githubusercontent.com/60442877/214995078-be024c66-a810-416a-9e87-b324cb7e3d16.png)

# Ranking Window Functions

![image](https://user-images.githubusercontent.com/60442877/214995144-b3e01168-d38f-4b23-a5b4-526461fbf582.png)
![image](https://user-images.githubusercontent.com/60442877/214995179-a94ee988-126c-4beb-a956-7d856bf953f1.png)
![image](https://user-images.githubusercontent.com/60442877/214995208-5f0793b7-ffed-4b1f-8b98-e1236e5ed1b7.png)

# MS SQL Server LEAD and LAG function

## LAG function

![image](https://user-images.githubusercontent.com/60442877/217402708-2883d820-3ba2-44b9-8cd6-c6805f335411.png)
![image](https://user-images.githubusercontent.com/60442877/217402745-9d6051cf-a74d-4e89-a0f5-795b79a2996d.png)
![image](https://user-images.githubusercontent.com/60442877/217402813-6dc84481-37b8-458e-adc3-47e3a25613cb.png)
![image](https://user-images.githubusercontent.com/60442877/217403066-b2bbee8d-535d-417c-82f1-710efd32e389.png)
![image](https://user-images.githubusercontent.com/60442877/217403079-3912c7fa-ee97-4e14-8fc4-a2c197a1473f.png)

## LEAD function

![image](https://user-images.githubusercontent.com/60442877/217403185-4c63ec61-ea25-480f-845e-f5b7907f1731.png)
![image](https://user-images.githubusercontent.com/60442877/217403203-f09cccee-4d40-496c-b4ca-d9d48b2da5c8.png)
![image](https://user-images.githubusercontent.com/60442877/217403227-54a35a9f-4a57-47b4-aaae-4571ade1536a.png)
![image](https://user-images.githubusercontent.com/60442877/217403253-8089b6c3-fd70-42dc-ab9e-c999c0b99cbd.png)
![image](https://user-images.githubusercontent.com/60442877/217403282-d0b13d5f-5bdb-421b-a85b-5641b0214637.png)


# Leetcode Question

## Dense_Rank() 
![image](https://user-images.githubusercontent.com/60442877/213212657-15b5f8ff-a627-489e-ad29-a4721de35f8f.png)
![image](https://user-images.githubusercontent.com/60442877/213212698-979839c8-cba4-4c95-94e7-6bcadda1513f.png)

    select score, Dense_Rank() over(order by score DESC) as 'rank'
    from Scores
    order by score Desc

## Dense_Rank() 
![image](https://user-images.githubusercontent.com/60442877/213300902-54be024a-019c-4076-869b-a9ae7c4ac4d4.png)
![image](https://user-images.githubusercontent.com/60442877/213300952-67dd5f19-6742-4cd2-b1aa-7989c2650a30.png)
![image](https://user-images.githubusercontent.com/60442877/213300992-3d1bbf75-ec6d-4520-9585-0c873dc3015e.png)

    select D.name as Department, P1.name as Employee, P1.salary as Salary
    from (select name, salary, departmentId, dense_rank() over(partition by departmentId order by salary DESC) AS emp_dense_rank from Employee) P1 join Department D on P1.departmentId = D.id
    where P1.emp_dense_rank in (1,2,3)

## Cumulative Sum
![image](https://user-images.githubusercontent.com/60442877/213603725-272fbce1-2166-4910-bb6e-90e735bcd498.png)
![image](https://user-images.githubusercontent.com/60442877/213603742-a7cc6735-b543-4a90-a611-fa38ab9792e6.png)

    select player_id, event_date, sum(games_played) over(partition by player_id order by event_date) as games_played_so_far
    from Activity

## Row_number() and Count()
![image](https://user-images.githubusercontent.com/60442877/213952664-f475bf6c-c16b-4959-8ee1-2b143009ab0a.png)
![image](https://user-images.githubusercontent.com/60442877/213952684-26eb223d-6616-423a-8c8b-61ed2304f855.png)
![image](https://user-images.githubusercontent.com/60442877/213952730-0f1ed531-17fa-4fd7-9173-a23ef5dcb23d.png)

    with temp1 as(
    select id, company, salary, row_number() over (partition by company order by salary,id) as emp_row_no
    from Employee),
    temp2 as(
    select company, count(salary) as salary_number from Employee group by company
    ),
    temp3 as(
        select company, if(salary_number % 2 != 0, (salary_number+1)/2, salary_number/2) as median_numb from temp2
        union
        select company, if(salary_number % 2 != 0, (salary_number+1)/2, salary_number/2+1) as median_numb from temp2
    )
    
    select id, company, salary
    from temp1
    where (company, emp_row_no) in (select * from temp3)

#################################################################################

    select Id, Company, Salary
    from (select id, company, salary, 
           row_number() over(partition by company order by salary, id) salaryrank, 
           count(*) over(partition by company) tte 
        from employee) as foo
    where salaryrank >= tte/2 and salaryrank <= tte/2+1

# Cumulative Sum and Total Sum

![image](https://user-images.githubusercontent.com/60442877/214458582-d64ece44-a014-433f-b7f3-5c1af28d6358.png)
![image](https://user-images.githubusercontent.com/60442877/214458591-ca227946-8a7d-4165-9598-e250d4e0567c.png)
![image](https://user-images.githubusercontent.com/60442877/214458533-f4c090f2-13bb-43b7-ad30-1452546326cd.png)

    with t as (select *, sum(frequency) over(order by number) freq, (sum(frequency) over())/2 median_num
               from numbers)

    select avg(number) as median
    from t
    where median_num between (freq-frequency) and freq


# Cumulative Calculation with preceding or following range

## Leetcode question 1 (Cumulative Sum within a specified range)

![image](https://user-images.githubusercontent.com/60442877/215009197-b7511934-14e9-4e7e-b075-db86dfaa8bf9.png)
![image](https://user-images.githubusercontent.com/60442877/215009252-ae6d1f04-ca6c-49e7-a2a5-c440e25d9a5c.png)
![image](https://user-images.githubusercontent.com/60442877/215009262-299006e1-9705-405c-ba13-e5ccaa0b9806.png)

    with cte as (
        select id, month, 
            sum(salary) over(partition by id order by month range between 2 preceding and current row) as 'Salary',
            rank() over(partition by id order by month desc) as rk
        from Employee
    )
    select id, month, Salary
    from cte 
    where rk > 1
    
    # consecutive calculation by month, for example, current row month is 7, precieding 2 month should be 6 and 5, even though those 2 months are not existing in the table
    
## LeetCode Question 2 (consecutive id problem)

![image](https://user-images.githubusercontent.com/60442877/217099861-0f186a35-b55c-4646-b81a-99245af5a044.png)
![image](https://user-images.githubusercontent.com/60442877/217099879-5995f479-e792-4d30-bbdc-2ba6c35c1bfe.png)

    with q1 as (
    select *, 
         count(*) over( order by id range between current row and 2 following ) following_cnt,
         count(*) over( order by id range between 2 preceding and current row ) preceding_cnt,
         count(*) over( order by id range between 1 preceding and 1 following ) current_cnt
    from stadium
    where people > 99
    )
    select id, visit_date, people
    from q1
    where following_cnt = 3 or preceding_cnt = 3 or current_cnt = 3
    order by visit_date
    
    # consecutive calculation by id, for example, if current row is id 3, following two rows should be 4 and 5 even though id 4 not exist for people > 99
    
###########################################################################

    with q1 as (
    select *, id - row_number() over() as id_diff
    from stadium
    where people > 99
    )
    select id, visit_date, people
    from q1
    where id_diff in (select id_diff from q1 group by id_diff having count(*) > 2)
    order by visit_date

# Consecutive id problem 

![image](https://user-images.githubusercontent.com/60442877/217401449-076cc851-917e-4a6b-9034-212e29ed9b6b.png)
![image](https://user-images.githubusercontent.com/60442877/217401464-b078ba46-45f6-4400-8dea-9532f42f7665.png)

    with temp1 as(
    select *, seat_id - row_number() over(order by seat_id) as temp_numb
    from Cinema
    where free = 1),
    temp2 as(
        select *, count(*) over(partition by temp_numb) as temp_numb2
        from temp1
    )

    select seat_id
    from temp2
    where temp_numb2 > 1

# Row Swap Problem

![image](https://user-images.githubusercontent.com/60442877/219551289-e1b58fd8-173b-44a5-8cfe-8907c8fbe893.png)
![image](https://user-images.githubusercontent.com/60442877/219551304-b59a162f-7bfc-430b-8d3b-595b5f27d988.png)


    select row_number() over(order by new_id) as id, student
    from (select id + if(id % 2 = 1, 1, -1) as new_id, student from Seat) as tb





